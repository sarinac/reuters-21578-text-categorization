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

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """Conduct forward pass."""
        x = self.embedding(x)  # out: batch_size, seq_length, embedding_dim
        x = torch.transpose(x, 0, 1)  # Same thing as LSTM's batch_first=True but more efficient?
        x, (h, c) = self.lstm(x)  # out: seq_length, batch_size, hidden_dim
        x = self.linear(h[-1])  # in: batch_size, hidden_dim; out: batch_size, 1
        return self.activation(x.squeeze())
