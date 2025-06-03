import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_fn_str="relu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        
        if activation_fn_str.lower() == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn_str.lower() == "silu":
            self.activation_fn = nn.SiLU()
        else: # Default to ReLU
            self.activation_fn = nn.ReLU()
            
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states, **kwargs): # Accept arbitrary kwargs like position_ids
        x = self.fc1(hidden_states)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x # Returns a tensor (batch_size, seq_len, hidden_size)