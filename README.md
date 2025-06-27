LSTM Tagger - Sequence Labeling with PyTorch
This project is a basic implementation of a tagger using LSTM (Long Short-Term Memory networks) in PyTorch. The goal is to label each word in a sentence with a tag, similar to how parts of speech tagging works.

What this project includes
- A small dataset of example sentences, each tagged with labels like BET, CN, and V.
  
- A function to convert words and tags into numerical format using index dictionaries.
  
- An LSTM model that:
  - Converts words into vector embeddings
  - Processes them through an LSTM layer
  - Outputs a prediction for the tag of each word

- A training loop that teaches the model using the small dataset

- Code to test the model before and after training

What I’ve learned so far
- How to build a basic LSTM model in PyTorch

- How to prepare text data for training by converting words into numbers

- How the embedding layer, LSTM, and linear layers work together

- How to train a model with loss functions and optimizers

- The importance of resetting the LSTM's hidden state during training

- How to interpret model predictions by comparing output scores

What can be done next
- Add more training data to improve the model’s accuracy

- Handle more realistic sentences using proper datasets

- Add batching and padding to train more efficiently

- Plot the loss over time to track learning progress
