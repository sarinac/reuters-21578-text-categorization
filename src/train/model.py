import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """Create a RNN model by setting up the layers.
        
        Parameters
        ----------
        vocab_size : int
            size of word dictionary
        embedding_dim: int
            size of embedding layer
        hidden_dim: int
            size of hidden layer
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.activation = nn.Sigmoid()
        

    def forward(self, x):
        """Conduct forward pass."""
        x = x.t()
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return self.activation(x)
