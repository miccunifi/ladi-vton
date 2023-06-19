import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


class InversionAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim:int, output_dim, config, num_encoder_layers, dropout=0.5):
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(num_encoder_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, None, None)
            x = x[0]
        x = x[:, 0, :]
        x = self.post_layernorm(x)
        return self.layers(x)
