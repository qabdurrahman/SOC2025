# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

%matplotlib inline

# %%
# training sentences and their corresponding word-tags
train_data = [
    ("The mouse ate the cheese".lower().split(), ["BET", "CN", "V", "BET", "CN"]),
    ("He read that book".lower().split(), ["CN", "V", "BET", "CN"]),
    ("The dog loves art".lower().split(), ["BET", "CN", "V", "CN"]),
    ("The elephant answers the phone".lower().split(), ["BET", "CN", "V", "BET", "CN"])
]

# create a dictionary that maps words to indices
word_to_index = {}

for sent, tags in train_data:
    for word in sent:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

# create a dictionary that maps tags to indices
tag_to_index = {"BET": 0, "CN": 1, "V": 2}

# %%
# print out the created dictionary
print(word_to_index)

# %%
import numpy as np

def encode_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    idxs = np.array(idxs)
    return torch.from_numpy(idxs)

# %%
example_input = encode_sequence("The mouse answers the phone".lower().split(), word_to_index)
print(example_input)

# %%
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_predictions = F.log_softmax(tag_outputs, dim=1)
        return tag_predictions

# %%
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

lstm_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(tag_to_index))

loss_fn = nn.NLLLoss()
opt = optim.SGD(lstm_model.parameters(), lr=0.1)

# %%
demo_sentence = "The book loves the elephant".lower().split()
sample_input = encode_sequence(demo_sentence, word_to_index)
tag_predictions = lstm_model(sample_input)
print(tag_predictions)
_, output_tags = torch.max(tag_predictions, 1)
print('\n')
print('Predicted tags: \n', output_tags)

# %%
n_epochs = 300
for epoch in range(n_epochs):
    batch_loss = 0.0
    for sentence, tags in train_data:
        lstm_model.zero_grad()
        lstm_model.hidden = lstm_model.init_hidden()
        input_tensor = encode_sequence(sentence, word_to_index)
        target_tensor = encode_sequence(tags, tag_to_index)
        tag_predictions = lstm_model(input_tensor)
        loss = loss_fn(tag_predictions, target_tensor)
        batch_loss += loss.item()
        loss.backward()
        opt.step()
    if(epoch % 20 == 19):
        print("Epoch: %d, loss: %1.5f" % (epoch+1, batch_loss/len(train_data)))

# %%
demo_sentence = "The mouse loves the elephant".lower().split()
sample_input = encode_sequence(demo_sentence, word_to_index)
tag_predictions = lstm_model(sample_input)
print(tag_predictions)
_, output_tags = torch.max(tag_predictions, 1)
print('\n')
print('Predicted tags: \n', output_tags)

