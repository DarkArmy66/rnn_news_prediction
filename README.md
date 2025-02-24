# RNN News Classification

This project, developed by Waldo Ungerer and Maedeh Amini, implements a Recurrent Neural Network (RNN) for the classification of news articles into four categories: world, business, sci-tech, and sports. The model is trained using PyTorch, with tokenization facilitated by Hugging Face Transformers, and the AG News dataset is employed for both training and evaluation.

## Usage
1. Input a news snippet into the interactive widget.
2. Click the "Predict" button.
3. The model predicts one of four categories: world, business, sci-tech, sports.

## Project Components

### 1. Data Processing
- The AG News dataset (from datasets library) is used.
- Each article is labeled with one of four categories.
- BERT-base-uncased is used for tokenization.
- The dataset is split into training (120,000 samples) and testing (7,600 samples).

**Tokenization:**
- Converts text into numerical format.
- Uses max sequence length of 128 tokens.

### 2. Model Architecture
- **Embedding Layer:** Converts tokens into dense vectors.
- **LSTM Layer:** Processes sequential data, capturing long-range dependencies.
- **Fully Connected Layer:** Outputs class probabilities using Softmax.
- **Loss Function:** CrossEntropyLoss is used as the loss function.
- **Optimizer:** Adam Optimizer is applied for training.

### 3. Training Process
- The model is trained for 5 epochs.
- Batch size: 32
- Learning rate: 0.001
- The best model checkpoint is saved as `rnn_model.pth`.

### 4. Model Evaluation
- Achieved 91.39% accuracy on the test dataset.
- Loss and accuracy trends are visualized in the notebook.

### 5. Prediction Pipeline
- Users input text in the interactive widget.
- Text is tokenized and converted to tensor format.
- The trained model predicts the most probable category.

## Model Performance
- **Training Accuracy:** 91.39%
- **Testing Accuracy:** 91.39%
- **Loss Reduction:** Steady decrease over epochs

## Future Work
- Implement bidirectional LSTMs for improved accuracy.
- Experimenting with GRUs (Gated Recurrent Units).
- Fine-tune pretrained transformers instead of training from scratch.
- Deploy as a REST API using FastAPI or Flask.
- Extend classification to more categories using larger datasets.

## References & Citations
- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS.
- Hugging Face's Transformers Library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- PyTorch Official Documentation: [https://pytorch.org/](https://pytorch.org/)
