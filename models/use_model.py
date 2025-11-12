import torch
import torch.nn as nn

class USE_Model(nn.Module):

  def __init__(self, use_dim, num_classes, multilabel=False, tda_dim=None):
    super().__init__()
    self.tda_dim = tda_dim
    self.multilabel = multilabel
    self.relu = nn.ReLU()

    if tda_dim is None:
        # Simple classifier for USE-only
        self.fc = nn.Sequential(
            nn.Linear(use_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    else:
        # Projections for USE and TDA features
        self.use_proj = nn.Sequential(
            nn.Linear(use_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.tda_proj = nn.Sequential(
            nn.Linear(tda_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Learnable feature-wise gating: α ∈ [0,1]^128
        self.gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.Sigmoid(),
        )

        # Fusion classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    if not multilabel:
        self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, use_feats, tda_feats=None):
    if self.tda_dim is None:
        x = self.fc(use_feats)
        return x if self.multilabel else self.softmax(x)

    # Project USE and TDA features
    use_out = self.use_proj(use_feats)
    tda_out = self.tda_proj(tda_feats)

    # Learn adaptive fusion per feature
    gate_input = torch.cat([use_out, tda_out], dim=1)
    alpha = self.gate(gate_input)  # shape [batch, 128]
    fusion = alpha * use_out + (1 - alpha) * tda_out

    # Combine and classify
    x = torch.cat((use_out, fusion), dim=1)
    x = self.fc(x)
    return x if self.multilabel else self.softmax(x)
