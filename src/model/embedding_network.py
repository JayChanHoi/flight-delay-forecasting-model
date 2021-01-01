import torch.nn as nn
import torch

class LearnedSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FTN(nn.Module):
    def __init__(self, input_dim=122, embedding_dim=32, dropout_p=0.1):
        super(FTN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4*embedding_dim, 3*embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(3*embedding_dim, 2*embedding_dim),
            LearnedSwish(),
            nn.Dropout(dropout_p),
            nn.Linear(2*embedding_dim, embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2*embedding_dim, 3 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(3*embedding_dim, 4 * embedding_dim),
            LearnedSwish(),
            nn.Dropout(dropout_p),
            nn.Linear(4*embedding_dim, input_dim),
        )

    def __call__(self, inputs):
        compressed_representation = self.encoder(inputs)
        x = self.decoder(compressed_representation)

        return x, compressed_representation