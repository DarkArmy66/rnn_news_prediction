import torch.nn as nn

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):  
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out