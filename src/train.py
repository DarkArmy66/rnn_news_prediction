import torch

# Function to train the model
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()  # Zero the parameter gradients
            input_ids = batch["input_ids"]
            labels = batch["label"]

            outputs = model(input_ids)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights

            total_loss += loss.item()  # Accumulate the loss

        # Print loss for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# If you need to save the trained model, uncomment the following line
# torch.save(model.state_dict(), "models/rnn_model.pth")

# If you need to load a trained model, use the following lines
# model.load_state_dict(torch.load("models/rnn_model.pth"))
# model.eval()  # Set the model to evaluation mode
