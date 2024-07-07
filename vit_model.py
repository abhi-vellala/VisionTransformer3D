import torch
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# import numpy as np
from positional_encodings.torch_encodings import PositionalEncodingPermute3D

class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.) -> None:
        super(Attention, self).__init__()
        self.head_dim = d_model//num_heads
        self.qkv_layer = torch.nn.Linear(d_model, 3*d_model, bias=False)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.softmax_layer = torch.nn.Softmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.linear_layer = torch.nn.Linear(d_model, d_model)

    def scaled_dot_product(self, q,k,v):
        d_k = k.size()[-1]
        scaled = torch.matmul(q, k.transpose(-1, -2))/(d_k**-0.5)
        attended = self.softmax_layer(scaled)
        attended = self.dropout_layer(attended)
        vhat = torch.matmul(attended, v)
        return attended, vhat
    
    def forward(self, x):
        x = self.layer_norm(x)
        batch_size, c, d, w, h = x.size()
        qkv = self.qkv_layer(x)
        # print(qkv.size())
        qkv = qkv.view(batch_size, self.num_heads, c, d, w, 3*self.head_dim)
        # print(qkv.size())
        q, k, v = qkv.chunk(3, dim=-1)
        # print(q.size(), k.size(), v.size()) 
        attended, vhat = self.scaled_dot_product(q, k, v)
        vhat = vhat.view(batch_size, c, d, w, h)
        out = self.linear_layer(vhat)
        out = self.dropout_layer(out)
        return out
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.) -> None:
        super(FeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, d_model), 
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads=8, drop_out=0.) -> None:
        super(TransformerBlock, self).__init__()
        self.attention_layer = Attention(d_model, num_heads, drop_out)
        self.feedforward_layer = FeedForward(d_model, hidden_dim, drop_out)
        # self.layers = torch.nn.ModuleList([Attention(d_model, num_heads, drop_out),
        #                                    FeedForward(d_model, hidden_dim, drop_out)])
    def forward(self, x):
        x = self.attention_layer(x) + x
        x = self.feedforward_layer(x) + x
        return x
    


    
class VisionTransformer3D(torch.nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, num_classes, device, trans_depth=12, drop_out=0.) -> None:
        super(VisionTransformer3D, self).__init__()
        self.d_model = d_model
        self.device = device
        self.pos_encode = PositionalEncodingPermute3D(d_model)
        self.transformers = torch.nn.ModuleList([])
        for _ in range(trans_depth):
            self.transformers.append(TransformerBlock(d_model, hidden_dim, num_heads, drop_out))
        # self.transformer = TransformerBlock(d_model, hidden_dim, num_heads, drop_out)
        self.dropout = torch.nn.Dropout(drop_out)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, num_classes)
        )


    def forward(self, x):
        pos = self.pos_encode(torch.randn(x.size(), device=self.device))
        # print(f'Positional encoding size: {pos.size()}')
        x += pos
        # print(f'X after adding pos: {x.size()}')
        x = self.dropout(x)
        # print(f'X after dropout: {x.size()}')
        for transformer in self.transformers:
            x = transformer(x)
        # print(f'X after transformer: {x.size()}')
        x = x.mean(dim=(-1,-2))
        # print(f'X after mean: {x.size()}')
        x = x.squeeze(1)
        # print(f'X after squeeze: {x.size()}')
        x = self.mlp_head(x)
        # print(f'X after mlp: {x.size()}')
        return x


        
        
        




if __name__ == "__main__":

    x = torch.randn(1, 1, 224, 224, 224)
    d_model = 224
    num_heads = 8
    hidden_dim = 768
    patch_size = 16
    stride = 16
    num_classes = 2
    image_size = 224
    trans_depth = 1
    # model = TransformerBlock(d_model, hidden_dim=896, num_heads=num_heads)
    # print(model(x).shape)
    model = VisionTransformer3D(d_model, hidden_dim, num_heads, num_classes, device='cpu',trans_depth=trans_depth)
    print(model(x))
    