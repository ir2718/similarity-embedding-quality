import torch.nn.functional as F
import torch.nn as nn
import torch
    
class CosineSimilarity(nn.Module):

    def forward(self, out1, out2):
        out_1_norm = F.normalize(out1, p=2.0, dim=-1)
        out_2_norm = F.normalize(out2, p=2.0, dim=-1)

        return (out_1_norm * out_2_norm).sum(dim=-1)
    
class DifferenceConcatenation(nn.Module):

    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 3, num_classes, bias=True)

    def forward(self, out1, out2):
        concatenation = torch.cat((out1, out2, torch.abs(out1 - out2)), dim=1)
        return self.linear(concatenation)