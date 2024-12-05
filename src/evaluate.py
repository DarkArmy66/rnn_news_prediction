import torch

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["label"]

            outputs = model(input_ids)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)  # Accumulate the total number of labels
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = 100 * correct / total  # Calculate accuracy
    print(f"Accuracy: {accuracy:.2f}%")  # Print accuracy
