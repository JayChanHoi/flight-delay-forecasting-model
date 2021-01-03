import torch
import torch.nn as nn

class LearnedSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PredNetwork(nn.Module):
    def __init__(self, input_dim=122, embedding_dim=32, dropout_p=0.1):
        super(PredNetwork, self).__init__()
        self.discrete_feature_layer = nn.Sequential(
            nn.Linear(116, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.BatchNorm1d(2*embedding_dim, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * embedding_dim, 3 * embedding_dim),
            nn.BatchNorm1d(3*embedding_dim, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(3 * embedding_dim, 4 * embedding_dim),
            # nn.BatchNorm1d(4 * embedding_dim, affine=False),
            nn.ReLU()
        )

        self.continuous_feature_layer = nn.Sequential(
            nn.Linear(6, (embedding_dim//2)),
            nn.BatchNorm1d(embedding_dim//2, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear((embedding_dim//2), 2 * (embedding_dim//2)),
            nn.BatchNorm1d(2 * (embedding_dim//2), affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * (embedding_dim//2), 3 * (embedding_dim//2)),
            nn.BatchNorm1d(3 * (embedding_dim // 2), affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(3 * (embedding_dim//2), 4 * (embedding_dim//2)),
            # nn.BatchNorm1d(4 * (embedding_dim//2), affine=False),
            nn.ReLU()
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(4 * ((embedding_dim//2) + embedding_dim), 3 * embedding_dim),
            # nn.BatchNorm1d(3 * embedding_dim, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(3 * embedding_dim, 2 * embedding_dim),
            # nn.BatchNorm1d(2 * embedding_dim, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.pred_layer = nn.Linear(embedding_dim, 2)

    def __call__(self, inputs):
        discrete_x = self.discrete_feature_layer(inputs[:, :116])
        continuous_x = self.continuous_feature_layer(inputs[:, 116:])
        fused_x = self.fusion_layer(torch.cat([discrete_x, continuous_x], dim=1))
        output = self.pred_layer(fused_x)

        return output