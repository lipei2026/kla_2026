import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import random
import pandas as pd

from group_splits import group_train_validation_split, load_fixed_group_folds


FOLD_REFERENCE_PATH = './train_data_with_ids.xlsx'
NUM_FOLDS = 5
SEED = 42
SPLIT_TAG = 'E4folds_grouped'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2, max_seq_length=51):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.position_embedding = nn.Embedding(max_seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.long()
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        word_embeddings = self.embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + position_embeddings
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(embeddings, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def seq2num(seqlist):
    out = []
    transdic = {'A': 8, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 0, 'L': 9, 'M': 10,
                'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '*': 20, 'B': 21}
    for seq in seqlist:
        seq = seq.replace('U', '*').replace('X', '*')
        vec = [transdic[i] for i in seq]
        out.append(vec)
    out = np.array(out)
    return out

def load_dataset(file_path):
    # Refuse to regenerate folds: LSTM must use E4's persisted protein folds.
    df = load_fixed_group_folds(file_path, n_splits=NUM_FOLDS, seed=SEED)
    sequences = df['Sequence'].tolist()
    labels = df['Label'].tolist()
    sequences = seq2num(sequences)
    sequences = np.array([np.array(seq) for seq in sequences])
    labels = np.array(labels)

    return sequences, labels, df


set_seed(SEED)
file_path = FOLD_REFERENCE_PATH
sequences_sum, labels_sum, metadata = load_dataset(file_path)
vocab_size = 22
num_sf = NUM_FOLDS
fold = 1
all_best_pred_scores = []
all_true_labels = []
result_rows = []
os.makedirs('./trained_models/LSTM', exist_ok=True)

for fold in range(1, num_sf + 1):
    set_seed(SEED + fold)
    train_index = np.flatnonzero(metadata['Fold'].to_numpy() != fold)
    test_index = np.flatnonzero(metadata['Fold'].to_numpy() == fold)
    inner_train_index, val_index = group_train_validation_split(
        labels_sum[train_index],
        metadata.iloc[train_index]['Group_ID'].to_numpy(),
        validation_size=0.1,
        seed=SEED + fold,
    )
    validation_index = train_index[val_index]
    train_index = train_index[inner_train_index]

    X_train, X_test = sequences_sum[train_index], sequences_sum[test_index]
    y_train, y_test = labels_sum[train_index], labels_sum[test_index]
    X_val, y_val = sequences_sum[validation_index], labels_sum[validation_index]

    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    test_dataset = ProteinDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    input_size = 128
    hidden_size = 256
    num_layers = 6
    num_classes = 2
    dropout_rate=0.3
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes,dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_auc = -np.inf
    model_path = f'./trained_models/LSTM/fold_{fold}_{num_sf}_{SPLIT_TAG}_lstm.pth'
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        validation_predictions = []
        validation_labels = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                validation_predictions.extend(probs)
                validation_labels.extend(labels.cpu().numpy())
        auc = roc_auc_score(validation_labels, validation_predictions)
        print(
            f'Fold {fold}, Epoch {epoch + 1}/{num_epochs}, '
            f'Loss: {running_loss / len(train_loader)}, Validation AUC: {auc:.6f}'
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    best_pred_scores = []
    with torch.no_grad():
        for batch_sequences, _ in test_loader:
            outputs = model(batch_sequences.to(device))
            best_pred_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    all_best_pred_scores.extend(best_pred_scores)
    all_true_labels.extend(y_test)
    result_rows.extend(
        {
            'Sample_ID': metadata.iloc[index]['Sample_ID'],
            'Group_ID': metadata.iloc[index]['Group_ID'],
            'Fold': fold,
            'label': int(labels_sum[index]),
            'score': float(score),
        }
        for index, score in zip(test_index, best_pred_scores)
    )

result_df = pd.DataFrame(result_rows)
print(roc_auc_score(all_true_labels,all_best_pred_scores))
result_df.to_csv(
    f"./scores/LSTM_{SPLIT_TAG}_y_label&score{num_sf}.csv",
    index=False,
)
