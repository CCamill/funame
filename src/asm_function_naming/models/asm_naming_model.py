"""
ASM Function Naming Model

核心模型：结合CLAP-ASM编码器、投影层和Qwen2.5-Coder
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Optional, Dict, Any, Tuple, List
import warnings

from .clap_asm import CLAPAsmEncoder
from .projection import AsmProjection, create_projection


class AsmNamingModel(nn.Module):
    """
    ASM函数命名模型
    
    架构：
    ASM Code -> CLAP-ASM -> Projection -> Prefix Tokens
                                              |
                                              v
    Prompt -> Qwen Tokenizer -> [Prefix | Prompt Tokens] -> Qwen -> Function Name
    """
    
    def __init__(
        self,
        clap_model_name: str = "hustcw/clap-asm",
        qwen_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        clap_hidden_size: int = 768,
        qwen_hidden_size: int = 2048,
        num_prefix_tokens: int = 32,
        projection_type: str = "mlp",
        use_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        freeze_clap: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.num_prefix_tokens = num_prefix_tokens
        self.qwen_hidden_size = qwen_hidden_size
        
        # 1. 加载CLAP-ASM编码器
        print("Loading CLAP-ASM encoder...")
        self.clap_encoder = CLAPAsmEncoder(
            model_name_or_path=clap_model_name,
            hidden_size=clap_hidden_size,
            pooling_strategy="cls",
            freeze=freeze_clap
        )
        
        # 2. 创建投影层
        print("Creating projection layer...")
        self.projection = create_projection(
            projection_type=projection_type,
            clap_hidden_size=clap_hidden_size,
            llm_hidden_size=qwen_hidden_size,
            num_prefix_tokens=num_prefix_tokens
        )
        
        # 3. 加载Qwen模型（4bit量化）
        print("Loading Qwen model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            trust_remote_code=True
        )
        
        # 确保有pad token
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            
        # 量化配置
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
            
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 4. 添加LoRA
        if use_lora:
            print("Adding LoRA adapters...")
            if use_4bit:
                self.qwen = prepare_model_for_kbit_training(self.qwen)
                
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.qwen = get_peft_model(self.qwen, lora_config)
            self.qwen.print_trainable_parameters()
            
        # 5. 创建用于prefix的虚拟embedding层（用于获取位置编码等）
        self.prefix_tokens_placeholder = nn.Parameter(
            torch.zeros(1, num_prefix_tokens, qwen_hidden_size),
            requires_grad=False
        )
        
    def get_prefix_embeddings(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        获取ASM的prefix embeddings
        
        Args:
            asm_input_ids: CLAP tokenizer的input_ids
            asm_attention_mask: CLAP tokenizer的attention_mask
            
        Returns:
            prefix_embeds: [batch_size, num_prefix_tokens, qwen_hidden_size]
        """
        # CLAP编码
        asm_embedding = self.clap_encoder(
            input_ids=asm_input_ids,
            attention_mask=asm_attention_mask
        )
        
        # 投影到prefix tokens
        prefix_embeds = self.projection(asm_embedding)
        
        return prefix_embeds
    
    def forward(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            asm_input_ids: CLAP的input_ids [batch, asm_seq_len]
            asm_attention_mask: CLAP的attention_mask [batch, asm_seq_len]
            prompt_input_ids: Qwen的input_ids [batch, prompt_seq_len]
            prompt_attention_mask: Qwen的attention_mask [batch, prompt_seq_len]
            labels: 标签（函数名的token ids）[batch, prompt_seq_len]
            
        Returns:
            包含loss和logits的字典
        """
        batch_size = asm_input_ids.size(0)
        
        # 1. 获取prefix embeddings
        prefix_embeds = self.get_prefix_embeddings(asm_input_ids, asm_attention_mask)
        # [batch, num_prefix, hidden]
        
        # 2. 获取prompt的embeddings
        prompt_embeds = self.qwen.get_input_embeddings()(prompt_input_ids)
        # [batch, prompt_len, hidden]
        
        # 3. 拼接prefix和prompt embeddings
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        # [batch, num_prefix + prompt_len, hidden]
        
        # 4. 创建完整的attention mask
        prefix_attention_mask = torch.ones(
            batch_size, self.num_prefix_tokens,
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device
        )
        full_attention_mask = torch.cat([prefix_attention_mask, prompt_attention_mask], dim=1)
        
        # 5. 处理labels
        if labels is not None:
            # 为prefix位置创建-100标签（不计算loss）
            prefix_labels = torch.full(
                (batch_size, self.num_prefix_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None
            
        # 6. 通过Qwen
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss if outputs.loss is not None else None,
            "logits": outputs.logits,
            "prefix_embeds": prefix_embeds
        }
    
    @torch.no_grad()
    def generate(
        self,
        asm_code: str,
        prompt: str = "Based on the assembly code above, the function name is:",
        max_new_tokens: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
        num_beams: int = 1
    ) -> str:
        """
        生成函数名
        
        Args:
            asm_code: 汇编代码字符串
            prompt: 提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样
            do_sample: 是否采样
            num_beams: beam search宽度
            
        Returns:
            生成的函数名
        """
        self.eval()
        
        # 1. 编码ASM
        asm_inputs = self.clap_encoder.tokenizer(
            asm_code,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        asm_inputs = {k: v.to(self.device) for k, v in asm_inputs.items()}
        
        # 2. 获取prefix embeddings
        prefix_embeds = self.get_prefix_embeddings(
            asm_inputs["input_ids"],
            asm_inputs["attention_mask"]
        )
        
        # 3. 编码prompt
        prompt_inputs = self.qwen_tokenizer(
            prompt,
            return_tensors="pt"
        )
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        
        prompt_embeds = self.qwen.get_input_embeddings()(prompt_inputs["input_ids"])
        
        # 4. 拼接
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        
        # 5. Attention mask
        batch_size = 1
        prefix_attention_mask = torch.ones(
            batch_size, self.num_prefix_tokens,
            dtype=torch.long,
            device=self.device
        )
        full_attention_mask = torch.cat(
            [prefix_attention_mask, prompt_inputs["attention_mask"]], 
            dim=1
        )
        
        # 6. 生成
        outputs = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.qwen_tokenizer.pad_token_id,
            eos_token_id=self.qwen_tokenizer.eos_token_id
        )
        
        # 7. 解码（只取新生成的部分）
        # 注意：generate返回的是从inputs_embeds位置开始的所有tokens
        # 我们需要跳过prompt的长度
        prompt_len = prompt_inputs["input_ids"].size(1)
        generated_ids = outputs[0][prompt_len:]
        
        function_name = self.qwen_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()
        
        # 清理函数名（只保留第一个单词/标识符）
        function_name = self._clean_function_name(function_name)
        
        return function_name
    
    def _clean_function_name(self, name: str) -> str:
        """清理生成的函数名"""
        # 移除常见的前缀/后缀
        name = name.strip()
        
        # 只保留有效的标识符字符
        import re
        match = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', name)
        if match:
            return match.group(0)
        return name.split()[0] if name else "unknown"
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存各组件
        # 1. CLAP encoder
        torch.save(
            self.clap_encoder.state_dict(),
            os.path.join(save_path, "clap_encoder.pt")
        )
        
        # 2. Projection
        torch.save(
            self.projection.state_dict(),
            os.path.join(save_path, "projection.pt")
        )
        
        # 3. Qwen LoRA
        self.qwen.save_pretrained(os.path.join(save_path, "qwen_lora"))
        
        # 4. Tokenizers
        self.clap_encoder.tokenizer.save_pretrained(
            os.path.join(save_path, "clap_tokenizer")
        )
        self.qwen_tokenizer.save_pretrained(
            os.path.join(save_path, "qwen_tokenizer")
        )
        
        print(f"Model saved to {save_path}")
        
    @classmethod
    def load_pretrained(
        cls,
        load_path: str,
        device: str = "cuda"
    ) -> "AsmNamingModel":
        """加载预训练模型"""
        import os
        from peft import PeftModel
        
        # 创建基础模型（不加载权重）
        model = cls(
            use_lora=False,  # 先不加载LoRA
            device=device
        )
        
        # 加载各组件
        # 1. CLAP encoder
        model.clap_encoder.load_state_dict(
            torch.load(os.path.join(load_path, "clap_encoder.pt"), map_location=device)
        )
        
        # 2. Projection
        model.projection.load_state_dict(
            torch.load(os.path.join(load_path, "projection.pt"), map_location=device)
        )
        
        # 3. Qwen LoRA
        model.qwen = PeftModel.from_pretrained(
            model.qwen,
            os.path.join(load_path, "qwen_lora")
        )
        
        return model


class AsmNamingModelForAlignment(nn.Module):
    """
    用于对比学习对齐阶段的模型
    
    只包含CLAP encoder和投影层，用于与Qwen的中间表示对齐
    """
    
    def __init__(
        self,
        clap_model_name: str = "microsoft/codebert-base",
        qwen_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        clap_hidden_size: int = 768,
        qwen_hidden_size: int = 2048,
        projection_dim: int = 512,
        use_4bit: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.projection_dim = projection_dim
        
        # CLAP encoder + 投影头
        self.clap_encoder = CLAPAsmEncoder(
            model_name_or_path=clap_model_name,
            hidden_size=clap_hidden_size
        )
        
        self.clap_projection = nn.Sequential(
            nn.Linear(clap_hidden_size, clap_hidden_size),
            nn.GELU(),
            nn.Linear(clap_hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Qwen (冻结，只用于提取特征)
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
            
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            output_hidden_states=True
        )
        
        # 冻结Qwen
        for param in self.qwen.parameters():
            param.requires_grad = False
            
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            trust_remote_code=True
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            
        # Qwen特征投影头
        self.qwen_projection = nn.Sequential(
            nn.Linear(qwen_hidden_size, qwen_hidden_size // 2),
            nn.GELU(),
            nn.Linear(qwen_hidden_size // 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # log(1/0.07)
        
    def encode_asm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """编码ASM代码"""
        embedding = self.clap_encoder(input_ids, attention_mask)
        projected = self.clap_projection(embedding)
        projected = nn.functional.normalize(projected, p=2, dim=-1)
        return projected
    
    @torch.no_grad()
    def encode_src(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int = 16  # 使用第16层的输出
    ) -> torch.Tensor:
        """编码源代码（从Qwen的中间层提取特征）"""
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取指定层的hidden states
        hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden]
        
        # 使用mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 投影
        projected = self.qwen_projection(pooled.float())
        projected = nn.functional.normalize(projected, p=2, dim=-1)
        
        return projected
    
    def forward(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: torch.Tensor,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 计算对比学习损失
        """
        # 编码
        asm_embeds = self.encode_asm(asm_input_ids, asm_attention_mask)
        src_embeds = self.encode_src(src_input_ids, src_attention_mask)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_asm = logit_scale * asm_embeds @ src_embeds.t()
        logits_per_src = logits_per_asm.t()
        
        # InfoNCE损失
        batch_size = asm_embeds.size(0)
        labels = torch.arange(batch_size, device=asm_embeds.device)
        
        loss_asm = nn.functional.cross_entropy(logits_per_asm, labels)
        loss_src = nn.functional.cross_entropy(logits_per_src, labels)
        loss = (loss_asm + loss_src) / 2
        
        # 计算正负样本的余弦相似度用于监控
        with torch.no_grad():
            # 正样本相似度（对角线）
            pos_sim = (asm_embeds * src_embeds).sum(dim=-1)
            
            # 负样本相似度（非对角线）
            sim_matrix = asm_embeds @ src_embeds.t()
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
            neg_sim = sim_matrix[mask]
            
        return {
            "loss": loss,
            "logits_per_asm": logits_per_asm,
            "logits_per_src": logits_per_src,
            "pos_similarity": pos_sim.mean(),
            "neg_similarity": neg_sim.mean(),
            "asm_embeds": asm_embeds,
            "src_embeds": src_embeds
        }
