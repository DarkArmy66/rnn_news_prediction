import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Custom dataset class for AG News
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

# Function to load data
def load_data(batch_size=32, max_len=128):
    ag_news = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Preprocess function to tokenize the text
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)

    tokenized_ag_news = ag_news.map(preprocess_function, batched=True)

    # Create train and test datasets
    train_dataset = AGNewsDataset(tokenized_ag_news["train"])
    test_dataset = AGNewsDataset(tokenized_ag_news["test"])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
