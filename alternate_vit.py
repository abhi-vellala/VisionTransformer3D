import torch
import torch.nn as nn

# Custom
class PatchEmbedding3D(torch.nn.Module):
    def __init__(self, patch_size, stride, image_size):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.layer_norm = torch.nn.LayerNorm(image_size)
        self.proj = torch.nn.Linear(image_size, image_size)

    def extract_patches_3d(self, images, patch_size, stride):
        if isinstance(patch_size, int): 
            patch_size = (patch_size, patch_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        batch_size, c, h, w, d = images.size()
        patches = images.unfold(2, patch_size[0], stride[0]) \
                        .unfold(3, patch_size[1], stride[1]) \
                        .unfold(4, d, d)
        patches = patches.contiguous().view(batch_size, c, -1, 
                                            patch_size[0], patch_size[1], d)
        patches = patches.permute(0, 2, 1, 3, 4, 5)
        return patches
    
    def forward(self, x):
        patches = self.extract_patches_3d(x, self.patch_size, self.stride)
        batch_size, num_patches, c, h, w, d = patches.size()
        patches = self.layer_norm(patches)
        patches = self.proj(patches)
        patches = patches.view(batch_size, c, d, d, d)
        patches = self.layer_norm(patches)
        return patches

# By GPT
class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, D, H, W)
        print(f'x after conv layer: {x.size()}')
        x = x.flatten(2)  # (batch_size, embed_dim, D*H*W)
        print(f'x after flatten: {x.size()}')
        x = x.transpose(1, 2)  # (batch_size, D*H*W, embed_dim)
        print(f'x after transpose: {x.size()}')
        return x
    
class VisionTransformer3D(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 3
        print(f'NUmber of patches: {num_patches}')
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        # self.linear_proj = nn.Linear((num_patches+1)*embed_dim, img_size**3, bias=False)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        print('Patch embedding')
        x = self.patch_embed(x)
        print(f'x after patch embeding: {x.size()}')
        batch_size = x.size(0)
        print(f'defined cls token: {self.cls_token.size()}')
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        print(f'cls token after expanding: {cls_tokens.size()}')
        x = torch.cat((cls_tokens, x), dim=1)
        print(f'x after cat with cls tokens: {x.size()}')
        
        x += self.pos_embed
        print(f'x after adding pos embed: {x.size()}')
        x = self.pos_drop(x)
        print(f'x after dropout: {x.size()}')
        # x = self.linear_proj(x.flatten())
        # x = x.view(4, 1, 224, 224, 224)
        
        print(f'CHECKING FOR X VIEW: {x.size()}')
        
        x = self.transformer(x)
        print(f'x after transformer: {x.size()}')
        cls_output = x[:, 0]
        print(f'cls_output: {cls_output.size()}')
        logits = self.head(cls_output)
        print(f'logits shape: {logits.size()}')
        return logits


# Example usage
img_size = 224
patch_size = 16
in_channels = 1
num_classes = 2
embed_dim = 768
depth = 12
num_heads = 12

model = VisionTransformer3D(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads)
images = torch.randn(4, 1, 224, 224, 224)
logits = model(images)
print(logits.shape)  # Expected shape: (batch_size, num_classes)