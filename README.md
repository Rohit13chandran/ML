import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Load the dataset
df = pd.read_csv('https://github.com/srivatsan88/YouTubeLI/blob/master/dataset/consumer_compliants.zip?raw=true', compression='zip', sep=',', quotechar='"')

# Preprocess the data
df['Consumer complaint narrative'].fillna('', inplace=True)  # Fill NaN values with empty strings
X_train, X_test = train_test_split(df, test_size=0.2, random_state=111)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', np.unique(df['Product']), df['Product'])
weights = {index: weight for index, weight in enumerate(class_weights)}

# Tokenizer function
def tokenize(text):
    return text.split()  # Simple whitespace tokenizer

# Custom Dataset Class for PyTorch
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = lambda x: [word_to_index[word] for word in tokenize(x) if word in word_to_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        return torch.tensor(tokens, dtype=torch.long), label  # Return tokenized text as tensor and label

# Create vocabulary from training data
all_words = set()
for text in X_train['Consumer complaint narrative']:
    all_words.update(tokenize(text))

# Create a mapping from words to indices
word_to_index = {word: idx + 1 for idx, word in enumerate(all_words)}  # Start indexing from 1

# Collate function for padding sequences
def collate_fn(batch):
    texts, labels = zip(*batch)  # Unzip into texts and labels
    texts_padded = pad_sequence([torch.tensor(t) for t in texts], batch_first=True)  # Pad sequences
    labels_tensor = torch.tensor(labels)  # Convert labels to tensor
    return texts_padded, labels_tensor

# Create datasets and dataloaders with padding using collate_fn
train_dataset = ComplaintDataset(X_train['Consumer complaint narrative'].values, X_train['Product'].values)
test_dataset = ComplaintDataset(X_test['Consumer complaint narrative'].values, X_test['Product'].values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Define an RNN-based model with LSTM and dropout for regularization
class ImprovedRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(ImprovedRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size)  # +1 for padding index
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Embed input tokens
        x, _ = self.lstm(x)  # Pass through LSTM
        x = x[:, -1]  # Get last hidden state (for sequence classification)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Instantiate model parameters
vocab_size = len(word_to_index)  # Size of vocabulary based on training data
embed_size = 128   # Size of embeddings
hidden_size = 64   # Size of hidden state in LSTM
output_size = len(np.unique(df['Product']))  # Number of unique classes

model = ImprovedRNNClassifier(vocab_size, embed_size, hidden_size, output_size)

# Define loss function and optimizer with class weights to handle imbalance
criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(weights.values()), dtype=torch.float32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (adjust epochs as necessary)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for texts, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(texts)  # Forward pass
        
        loss = criterion(outputs, labels)  # Calculate loss
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Evaluation loop (optional)
model.eval()
correct_predictions = 0

with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        
        _, predicted_labels = torch.max(outputs.data, 1)
        correct_predictions += (predicted_labels == labels).sum().item()

accuracy = correct_predictions / len(test_dataset)
print(f'Test Accuracy: {accuracy:.4f}')
