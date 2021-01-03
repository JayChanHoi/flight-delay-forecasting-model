import torch
import torch.nn as nn

class LearnedSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PredNetwork():
    def __init__(self, input_dim=122, embedding_dim=32, dropout_p=0.1):
        super(PredNetwork, self).__init__()
        self.feature_extraction_layers = nn.Sequential(
            nn.Linear(input_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * embedding_dim, embedding_dim),
            LearnedSwish(),
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.pred_layer = nn.Linear(embedding_dim, 2)

    def __call__(self, inputs):
        x = self.feature_extraction_layers(inputs)
        output = self.pred_layer(x)

        return output