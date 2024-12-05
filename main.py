from src.data_loader import load_data
from src.model import RNNModel
from src.train import train_model
from src.evaluate import evaluate_model
import torch

def main():
    # Load data
    train_dataloader, test_dataloader = load_data()

    # Initialize model, criterion, and optimizer
    vocab_size = 30522  # Size of the tokenizer's vocabulary
    embed_size = 128
    hidden_size = 128
    output_size = 4

    # Initialize the model
    model = RNNModel(vocab_size, embed_size, hidden_size, output_size)

    # Load the trained model weights
    model.load_state_dict(torch.load("models/rnn_model.pth"))
    model.eval()  # Set the model to evaluation mode

    # Define criterion and optimizer (though they won't be used here)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Evaluate the model (No need to train as we are loading pre-trained model)
    evaluate_model(model, test_dataloader)

if __name__ == "__main__":
    main()
