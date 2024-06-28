from transformers import ViTForImageClassification, ViTConfig, ViTModel
import torch

# Configure the Vision Transformer
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=224,  # Depth dimension as channels
    num_labels=2,     # Number of classes
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob = 0.1,
    hidden_act = "gelu"

)

print(config.num_labels)
# Initialize the model
# model = ViTForImageClassification(config)
model = ViTModel(config)
# print(model)
# print(model.config)
x = torch.randn((4, 224, 224, 224))
x = model(x)
x = x.last_hidden_state[:,0,:]
print(x.shape)
# x = x.flatten(-1).unsqueeze(0)
# print(x.shape)
x = torch.nn.Linear(config.hidden_size, out_features=2)(x)
print(x.shape)

model1 = ViTForImageClassification(config)
# print(model1)
x = torch.randn((4, 224, 224, 224))
print(model1(x).logits.shape)
# ViTConfig {
#   "attention_probs_dropout_prob": 0.0,
#   "encoder_stride": 16,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.0,
#   "hidden_size": 768,
#   "image_size": 224,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "model_type": "vit",
#   "num_attention_heads": 12,
#   "num_channels": 3,
#   "num_hidden_layers": 12,
#   "patch_size": 16,
#   "qkv_bias": true,
#   "transformers_version": "4.37.0.dev0"
# }