import torch.nn.functional as F
import torch.nn as nn
import torch
    
class CosineSimilarity(nn.Module):

    def forward(self, out1, out2):
        out_1_norm = F.normalize(out1, p=2.0, dim=-1)
        out_2_norm = F.normalize(out2, p=2.0, dim=-1)

        return (out_1_norm * out_2_norm).sum(dim=-1)