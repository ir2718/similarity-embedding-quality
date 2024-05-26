import torch.nn.functional as F
import torch.nn as nn
import torch

# class DotProductSimilarity(nn.Module):

#     def forward(self, out1, out2, pairwise):
#         return (out1 * out2).sum(dim=1)
    
class CosineSimilarity(nn.Module):

    def forward(self, out1, out2, pairwise):
        out_1_norm = F.normalize(out1, p=2.0, dim=-1)
        out_2_norm = F.normalize(out2, p=2.0, dim=-1)

        if pairwise:
            return torch.mm(out_1_norm, out_2_norm.T)

        return (out_1_norm * out_2_norm).sum(dim=-1)