# RNN News Classification

This repository contains a Recurrent Neural Network (RNN) for classifying news articles using the AG News dataset. The model is pre-trained and includes an interactive Jupyter Notebook widget for testing.

---

## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Usage](#usage)
- [Model Details](#model-details)
- [Repository Structure](#repository-structure)

---

## Installation

Easy way: 

Linux: 
'''sh
./setup.sh
'''

Windows: 
'''sh
setup.bat
'''


Hard Way:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/DarkArmy66/rnn_news_prediction.git
    cd rnn_news_prediction
    ```

2. **Create a virtual environment**:
    ```sh
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Start Jupyter Notebook**:
    ```sh
    jupyter notebook
    ```

5. **Open and run the demo notebook**:
    - Navigate to `notebooks/demo.ipynb`
    - Run all the cells to load the model and interact with the widget.

---

## Usage

- **Type** Choose a random news article, copy a paragraph and paste it into the widget.
- **Click** the "Predict" button to view the predicted news category 

---

## Model Details

- **Dataset**: The model is trained on the AG News dataset.
- **Model File**: The trained model is saved as `models/rnn_model.pth`.
- **Architecture**: The RNN architecture is defined in `src/model.py`.

---



