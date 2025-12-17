import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    带困难样本挖掘的InfoNCE损失函数
    
    困难负样本是那些与查询样本相似但实际上是负样本的例子，
    它们对模型学习更有帮助。
    """
    def __init__(self, hard_negtive_weight:int = 1, margin:int = 0.2):
        super().__init__()
        self.hard_negtive_weight = hard_negtive_weight
        self.margin = margin

    def forward(self, 
                asm_embeddings:torch.Tensor, 
                src_embeddings: torch.Tensor, 
                temperature:float = 0.05):
        batch_size = asm_embeddings.shape[0]
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        src_embeddings = F.normalize(src_embeddings, p=2, dim = 1)
        
        sim_matrix = torch.matmul(asm_embeddings, src_embeddings.T) / temperature
        
        pos_mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[~pos_mask].view(batch_size, -1)
        
        hard_neg_sim, _ = neg_sim.max(dim=1)
        
        labels = torch.arange(batch_size, device=asm_embeddings.device)
        
        loss_asm2src = F.cross_entropy(sim_matrix, labels)
        loss_src2asm = F.cross_entropy(sim_matrix.T, labels)
        
        loss_basic = (loss_asm2src + loss_src2asm) / 2
        
        hard_neg_penalty = F.relu(hard_neg_sim - pos_sim + self.margin / temperature).mean()
        
        total_loss = loss_basic + self.hard_negtive_weight * hard_neg_penalty
        
        with torch.no_grad():
            accuracy = (sim_matrix.argmax(dim=1) == labels).float().mean().item()
            mean_pos_sim = pos_sim.mean().item() * temperature
            mean_neg_sim = neg_sim.mean().item() * temperature
        
        metrics = {
            'accuracy': accuracy,
            'mean_pos_sim': mean_pos_sim,
            'mean_neg_sim': mean_neg_sim,
            'hard_neg_penalty': hard_neg_penalty.item()
        }
        
        return total_loss, metrics 