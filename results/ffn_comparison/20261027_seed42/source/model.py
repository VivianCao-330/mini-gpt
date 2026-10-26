import torch
import torch.nn as nn
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(vocab_size,d_model)*0.02)
    def forward(self, token_ids):
        return self.weight[token_ids]
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        mean_squared=torch.mean(x**2,dim=-1, keepdim=True)
        rms=torch.sqrt(mean_squared+self.eps)
        normalized=x/rms
        return self.gamma * normalized
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model=d_model
        self.n_heads=n_heads
        self.head_dim=d_model//n_heads
        self.W_q=nn.Linear(d_model,d_model,bias=False)
        self.W_k=nn.Linear(d_model,d_model,bias=False)
        self.W_v=nn.Linear(d_model,d_model,bias=False)
        self.W_o=nn.Linear(d_model,d_model,bias=False)
        self.rope = RoPE(self.head_dim)
        self.use_rope = use_rope
    def forward(self,x):
        B,T,C=x.shape
        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)
        Q=Q.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        K=K.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        V=V.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        if self.use_rope:
            Q = self.rope(Q)
            K = self.rope(K)
        scores=Q@K.transpose(-2,-1)/(self.head_dim**0.5)
        future_mask=torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores=scores.masked_fill(future_mask, float("-inf"))
        attention_weights=torch.softmax(scores, dim=-1)
        attention_output=attention_weights@V
        attention_output=attention_output.transpose(1,2).reshape(B,T,C)
        return self.W_o(attention_output)
class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim=head_dim
        indices=torch.arange(0,head_dim, 2).float()
        inv_freq=1.0/(base ** (indices/head_dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x):
        B, H, T, D=x.shape
        positions=torch.arange(T, device=x.device)
        angles = positions.reshape(T, 1) * self.inv_freq.reshape(1, D // 2)
        cos=torch.cos(angles)
        sin=torch.sin(angles)
        x_pairs=x.reshape(B,H,T,D//2,2)
        x1=x_pairs[...,0]
        x2=x_pairs[...,1]
        y1=x1*cos-x2*sin
        y2=x1*sin+x2*cos
        rotated=torch.stack([y1,y2],dim=-1)
        return rotated.reshape(B,H,T,D)
class SwiGLUFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,bias=False):
        super().__init__()
        d_hidden=(2*d_ff)//3
        self.gate_proj=nn.Linear(d_model,d_hidden,bias=bias)
        self.up_proj=nn.Linear(d_model,d_hidden,bias=bias)
        self.down_proj=nn.Linear(d_hidden,d_model,bias=bias)
    def forward(self,x):
        gate=nn.functional.silu(self.gate_proj(x))
        up=self.up_proj(x)
        hidden=gate*up
        return self.down_proj(hidden)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="relu", bias=True):
        super().__init__()
        if activation not in ("relu", "gelu", "silu"):
            raise ValueError("activation must be relu, gelu, or silu")
        self.activation = activation
        self.fc1=nn.Linear(d_model,d_ff,bias=bias)
        self.fc2=nn.Linear(d_ff,d_model,bias=bias)
    def forward(self,x):
        x=self.fc1(x)
        if self.activation == "relu":
            x=torch.relu(x)
        elif self.activation == "gelu":
            x=torch.nn.functional.gelu(x)
        else:
            x=torch.nn.functional.silu(x)
        x=self.fc2(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self,d_model, n_heads,d_ff, use_rope=True, activation="relu", ffn_bias=None):
        super().__init__()
        self.norm1=RMSNorm(d_model)
        self.attention=CausalSelfAttention(d_model,n_heads,use_rope)
        self.norm2=RMSNorm(d_model)
        if activation == "swiglu":
            self.ffn=SwiGLUFeedForward(d_model,d_ff,bias=False if ffn_bias is None else ffn_bias)
        else:
            self.ffn=FeedForward(d_model,d_ff,activation,bias=True if ffn_bias is None else ffn_bias)
    def forward(self,x):
        x=x+self.attention(self.norm1(x))
        x=x+self.ffn(self.norm2(x))
        return x
class GPT(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,d_ff,n_layers,use_rope=True,activation="relu",ffn_bias=None):
        super().__init__()
        self.embedding=TokenEmbedding(vocab_size,d_model)
        self.blocks=nn.ModuleList([TransformerBlock(d_model,n_heads,d_ff,use_rope,activation,ffn_bias) for _ in range(n_layers)])
        self.final_norm=RMSNorm(d_model)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
    def forward(self,token_ids):
        x=self.embedding(token_ids)
        for block in self.blocks:
            x=block(x)
        x=self.final_norm(x)
        logits=self.lm_head(x)
        return logits
