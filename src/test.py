# let's build a recurrent neural network to predict the next word in a sentence
# we'll use pytorch for this
# let's use the AG_NEWS dataset for this

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

class AGNewsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }
    
def load_data(batch_size=32, max_len=128):
    ag_news = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)

    tokenized_ag_news = ag_news.map(preprocess_function, batched=True)

    train_dataset = AGNewsDataset(tokenized_ag_news["train"])
    test_dataset = AGNewsDataset(tokenized_ag_news["test"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader

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
    
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            labels = batch["label"]

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["label"]

            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total:.4f}")

train_dataloader, test_dataloader = load_data()

vocab_size = 30522

model = RNNModel(vocab_size, 128, 128, 4)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_dataloader, criterion, optimizer, num_epochs=5)

evaluate_model(model, test_dataloader)

# The model is trained on the AG_NEWS dataset and evaluated on the test set. The accuracy is printed at the end of the evaluation.

