import torch
import torch.nn as nn

class USE_Model(nn.Module):
    """
    Unified USE model supporting optional TDA features.
    Architecture is consistent whether TDA is used or not,
    allowing fair comparison between USE-only and USE+TDA.
    Dropout is applied the same number of times in all cases.
    """

    def __init__(self, use_dim, num_classes, multilabel=False, tda_dim=None):
        super().__init__()
        self.tda_dim = tda_dim
        self.multilabel = multilabel
        self.relu = nn.ReLU()

        # ----- Projection of USE features -----
        self.use_proj = nn.Sequential(
            nn.Linear(use_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # first dropout
        )

        # ----- Optional TDA projection -----
        if tda_dim is not None:
            self.tda_proj = nn.Sequential(
                nn.Linear(tda_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            # Feature-wise gating for fusion
            self.gate = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.Sigmoid(),
            )

        # ----- Final classifier (same for both USE-only and USE+TDA) -----
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # second dropout
            nn.Linear(128, num_classes),
        )

    def forward(self, use_feats, tda_feats=None):
        # Project USE features
        use_out = self.use_proj(use_feats)

        if self.tda_dim is not None and tda_feats is not None:
            # Project TDA features
            tda_out = self.tda_proj(tda_feats)

            # Feature-wise gating fusion
            gate_input = torch.cat([use_out, tda_out], dim=1)
            alpha = self.gate(gate_input)  # [batch, 128]
            fusion = alpha * use_out + (1 - alpha) * tda_out
        else:
            # No TDA → use USE features as fusion
            fusion = use_out

        # Concatenate USE projection + fusion vector (always 256-D)
        x = torch.cat([use_out, fusion], dim=1)
        logits = self.fc(x)

        return logits  # raw logits
