import nltk
nltk.download('treebank')
nltk.download('brown')
nltk.download('conll2000')

from nltk.corpus import treebank, brown, conll2000
tagged_sentences = treebank.tagged_sents(tagset='universal') + \
                   brown.tagged_sents(tagset='universal') + \
                   conll2000.tagged_sents(tagset='universal')

sentences, sentence_tags = [], []

for s in tagged_sentences:
    words, tags = zip(*s)
    sentences.append(list(words))
    sentence_tags.append(list(tags))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sentences, sentence_tags, test_size=0.2)

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

word_tokenizer = Tokenizer(oov_token='<OOV>')
word_tokenizer.fit_on_texts(X_train)
X_train_seq = word_tokenizer.texts_to_sequences(X_train)

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(y_train)
y_train_seq = tag_tokenizer.texts_to_sequences(y_train)

MAXLEN = 170
X_train_pad = pad_sequences(X_train_seq, maxlen=MAXLEN, padding='pre')
y_train_pad = pad_sequences(y_train_seq, maxlen=MAXLEN, padding='pre')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PosDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PosDataset(X_train_pad, y_train_pad)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

vocab_size = len(word_tokenizer.word_index) + 1
num_classes = len(tag_tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 128

class POSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(POSTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = x != 0
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        output = self.fc(lstm_out)
        return output, mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = POSTagger(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output, mask = model(X_batch)
        loss = loss_fn(output.view(-1, num_classes), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def tag_sentences(sentences: list):
    model.eval()
    sequences = word_tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=MAXLEN, padding='pre')
    input_tensor = torch.LongTensor(padded).to(device)

    with torch.no_grad():
        output, _ = model(input_tensor)
        pred_tags = torch.argmax(output, dim=-1).cpu().numpy()

    result = []
    for i, seq in enumerate(sequences):
        tags_seq = pred_tags[i][-len(seq):]
        words = [word_tokenizer.index_word[idx] for idx in seq]
        tags = [tag_tokenizer.index_word[tag] for tag in tags_seq]
        result.append(list(zip(words, tags)))
    
    return result

samples = [
    "Brown refused to testify.",
    "Come as you are",
]
tagged = tag_sentences(samples)
for sent in tagged:
    print(sent)
