# Testing Custom ViT model

import torch

class PatchEmbedding3D(torch.nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.proj = torch.nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)  # Flatten the spatial dimensions
        x = x.transpose(1, 2)  # Rearrange to (batch_size, num_patches, emb_size)
        return x
    
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
        assert self.d_model == x.size(-1)
        batch_size, n, _ = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.view(batch_size, self.num_heads, n, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        attended, vhat = self.scaled_dot_product(q, k, v)
        vhat = vhat.view(batch_size, n, self.d_model)
        out = self.linear_layer(vhat)
        out = self.dropout_layer(out)
        return out, attended
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.) -> None:
        super(FeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, d_model), 
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, feedforward_dim, num_heads=8, drop_out=0.) -> None:
        super(TransformerBlock, self).__init__()
        self.attention_layer = Attention(d_model, num_heads, drop_out)
        self.feedforward_layer = FeedForward(d_model, feedforward_dim, drop_out)
 
    def forward(self, x):
        attn_out, attn_weights = self.attention_layer(x)
        x = x + attn_out
        x = x + self.feedforward_layer(x)
        return x, attn_weights
    
class VisionTransformer3D(torch.nn.Module):
    def __init__(self, in_channels, d_model, feedforward_dim, patch_size, num_classes, num_heads=8, num_layers=1, drop_out=0.):
        super(VisionTransformer3D, self).__init__()
        self.patch_embedding = PatchEmbedding3D(in_channels, patch_size, d_model)
        self.num_patches = (224//patch_size)**3
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        self.transformerlayers = torch.nn.ModuleList([TransformerBlock(d_model, feedforward_dim, num_heads, drop_out)
                                                      for _ in range(num_layers)])
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.last_linear_layer = torch.nn.Linear(d_model, num_classes)


    def forward(self, x):
        batch_size, c, h, w, d = x.size()
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        assert x[:,0].size() == cls_token[:,0].size()
        x += self.pos_embedding
        for layer in self.transformerlayers:
            x, attn_weights = layer(x)
        x = self.layer_norm(x)
        cls_logits = self.last_linear_layer(x[:,0])
        return cls_logits, attn_weights
    
def UpsampleAttentionMap(attention_map, size):
    attention_map = attention_map.unsqueeze(1)  # Add channel dimensions
    upsampled_attention_map = torch.nn.functional.interpolate(attention_map, size=size, mode='trilinear', align_corners=False)
    return upsampled_attention_map.squeeze(1)



# Example usage
if __name__ == '__main__':
    in_channels = 1
    d_model = 256
    feedforward_dim = 512
    num_heads = 8
    patch_size = 16
    num_layers = 12
    dropout = 0.1
    num_classes = 2
    batch_size = 3
    x = torch.randn(batch_size, in_channels, 224, 224, 224)  # Example 3D medical image 11239424
    
    model = VisionTransformer3D(in_channels, d_model, feedforward_dim, patch_size, num_classes=2, num_layers=num_layers)
    out, attn_weights = model(x)
    attention_map = attn_weights[:,:,0,1:]
    print(f'attention map: {attention_map.size()}')
    attention_map = attention_map.mean(1)
    print(f'attention map after mean: {attention_map.size()}')
    attention_map = attention_map.reshape(batch_size, 14, 14, 14)
    print(f'attention map after reshape: {attention_map.size()}')
    attention_map_upsampled = UpsampleAttentionMap(attention_map, (224, 224, 224))
    print(f'attention map after upsampling: {attention_map_upsampled.size()}')
    