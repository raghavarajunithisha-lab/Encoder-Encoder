import torch
import torch.nn as nn

class BERT_Arch(nn.Module):
    """
    Unified BERT/BART model with optional TDA features.
    Architecture is consistent whether TDA is used or not,
    allowing fair comparison between encoder-only and encoder+TDA.
    """

    def __init__(self, bert_model, num_classes, tda_dim=None, use_tda=False, multilabel=True):
        super().__init__()
        self.use_tda = use_tda
        self.multilabel = multilabel
        self.relu = nn.ReLU()

        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size

        # ----- Projection of BERT features -----
        self.bert_proj = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # first dropout
        )

        # ----- Optional TDA projection -----
        if self.use_tda:
            self.tda_proj = nn.Sequential(
                nn.Linear(tda_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                
            )
            self.gate = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.Sigmoid(),
            )

        # ----- Final classifier (same for both encoder-only and encoder+TDA) -----
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # second dropout
            nn.Linear(128, num_classes),
        )

    def forward(self, sent_id, mask, tda_feats=None):
        # Project BERT [CLS] embedding
        bert_outs = self.bert(sent_id, attention_mask=mask)
        cls_hs = bert_outs.last_hidden_state[:, 0]  # [batch, hidden_size]
        bert_out = self.bert_proj(cls_hs)

        if self.use_tda and tda_feats is not None:
            tda_out = self.tda_proj(tda_feats)
            gate_input = torch.cat([bert_out, tda_out], dim=1)
            alpha = self.gate(gate_input)
            fusion = alpha * bert_out + (1 - alpha) * tda_out
        else:
            fusion = torch.zeros_like(bert_out)

        x = torch.cat([bert_out, fusion], dim=1)
        logits = self.fc(x)
        return logits  # raw logits
