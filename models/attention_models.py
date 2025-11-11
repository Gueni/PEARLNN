import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head attention for sequence data"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(x)

class AttentionModel(nn.Module):
    """Transformer-based model for waveform analysis"""
    
    def __init__(self, input_dim, output_dim, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_layers)
        ])
        
        # Output
        self.output_layer = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Input projection
        x = self.input_proj(x)
        
        # Transformer layers
        for attn, norm1, ffn, norm2 in zip(self.attention_layers, self.norm_layers, self.ffn_layers, self.norm_layers):
            # Self-attention with residual
            attn_out = attn(x, x, x, mask)
            x = norm1(x + self.dropout(attn_out))
            
            # Feed-forward with residual
            ffn_out = ffn(x)
            x = norm2(x + self.dropout(ffn_out))
        
        # Global average pooling and output
        x = x.mean(dim=1)  # Average over sequence length
        return self.output_layer(x)