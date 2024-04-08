import torch.nn as nn
import torch

class MaxPooling(nn.Module):
    def __init__(self, starting_state):
        super().__init__()
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        hidden_state = hidden.hidden_states[self.starting_state]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        hidden_state_max = torch.max(hidden_state, dim=1)[0]
        return hidden_state_max

class CLSPooling(nn.Module):
    def __init__(self, starting_state):
        super().__init__()
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        hidden_state = hidden.hidden_states[self.starting_state]
        return hidden_state[:, 0, :]
    
class MeanPooling(nn.Module):
    def __init__(self, starting_state):
        super().__init__()
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        # model truncation
        hidden_state = hidden.hidden_states[self.starting_state]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        emb_sum = torch.sum(hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        emb_mean = emb_sum / sum_mask
        return emb_mean