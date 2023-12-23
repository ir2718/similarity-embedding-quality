import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch

class MeanSelfAttentionPooling(nn.Module):
    def __init__(self, config, starting_state):
        super().__init__()
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        hidden_states = torch.stack(hidden.hidden_states).permute(1,0,2,3)
        input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1).permute(0,2,1,3).expand(hidden_states.size()).float()
        emb_sum = torch.sum(hidden_states * input_mask_expanded, dim=-2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=-2), min=1e-9)
        mean = emb_sum / sum_mask

        res = F.softmax(mean.matmul(mean.transpose(1, 2)), dim=2).matmul(mean)
        return res[:, self.starting_state]

class MeanEncoderPooling(nn.Module):
    """Adds an encoder over the hidden states."""
    def __init__(self, config, starting_state):
        super().__init__()
        self.starting_state = starting_state
        self.mha = MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )

        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)

        if config.hidden_act == "gelu":
            self.hidden_act = nn.GELU()
        elif config.hidden_act == "relu":
            self.hidden_act = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size),
            self.hidden_act,
            nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size),
            nn.Dropout(p=self.hidden_dropout_prob)
        )

    def forward(self, hidden, attention_mask):
        hidden_states = torch.stack(hidden.hidden_states).permute(1,0,2,3)
        input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1).permute(0,2,1,3).expand(hidden_states.size()).float()
        emb_sum = torch.sum(hidden_states * input_mask_expanded, dim=-2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=-2), min=1e-9)
        mean = emb_sum / sum_mask

        out, _ = self.mha(mean, mean, mean, need_weights=False)
        out_drop = F.dropout(out, p=self.hidden_dropout_prob)
        intermediate = self.layer_norm(out_drop + mean)

        ffn_out = self.ffn(intermediate)
        final = self.layer_norm(ffn_out + intermediate)

        return final[:, self.starting_state]

class MaxPooling(nn.Module):
    def __init__(self, last_k_states, starting_state):
        super().__init__()
        self.last_k = last_k_states
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        )
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        last_k_hidden[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        last_k_max = torch.max(last_k_hidden, dim=2)[0]
        mean_of_max = last_k_max.mean(dim=0) 
        return mean_of_max

class CLSPooling(nn.Module):
    def __init__(self, last_k_states, starting_state):
        super().__init__()
        self.last_k = last_k_states
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        last_k_cls = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        )[:, :, 0, :]
        mean_of_cls = last_k_cls.mean(dim=0)
        return mean_of_cls
    
class MeanPooling(nn.Module):
    def __init__(self, last_k_states, starting_state):
        super().__init__()
        self.last_k = last_k_states
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        ).mean(dim=0, keepdim=True)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        emb_sum = torch.sum(last_k_hidden * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=2), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean.squeeze(0)

class NormMeanPooling(nn.Module):
    def __init__(self, last_k_states, starting_state):
        super().__init__()
        self.last_k = last_k_states
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        ).mean(dim=0, keepdim=True)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        emb_sum = torch.sum(F.normalize(last_k_hidden * input_mask_expanded, p=2, dim=-1), dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=2), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean.squeeze(0)
    
class WeightedMeanPooling(nn.Module):
    def __init__(self, config, last_k, starting_state, init=None):
        super().__init__()
        self.last_k = last_k
        self.starting_state = starting_state
        self.mean_weights = torch.nn.Parameter(torch.ones(self.last_k, 1, 1, config.hidden_size), requires_grad=True)

    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        ).mean(dim=0, keepdim=True)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        emb_sum = torch.sum(last_k_hidden * self.mean_weights * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=2), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean.squeeze(0)
    
class WeightedPerComponentMeanPooling(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.mean_weights = torch.nn.Parameter(torch.randn(tokenizer.model_max_length, config.hidden_size))
        self.mean_weights.requires_grad = True


    def forward(self, hidden, attention_mask):
        last_hidden = hidden.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        emb_sum = torch.sum(last_hidden * self.mean_weights[:last_hidden.shape[1]] * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean