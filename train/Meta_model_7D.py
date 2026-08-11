import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import chain
from group_splits import group_train_validation_split, load_fixed_group_folds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOLD_REFERENCE_PATH = './train_data_with_ids.xlsx'
NUM_FOLDS = 5
SEED = 42
SPLIT_TAG = 'E4folds_grouped'
BASE_SCORE_FILES = {
    'E4': f'ESM2_E4_center_k_mean_grouped_y_label&score{NUM_FOLDS}.csv',
    'LSTM': f'LSTM_{SPLIT_TAG}_y_label&score{NUM_FOLDS}.csv',
    'ACF': f'ACF_{SPLIT_TAG}_y_label&score{NUM_FOLDS}.csv',
    'AAINDEX': f'AAINDEX_{SPLIT_TAG}_y_label&score{NUM_FOLDS}.csv',
    'CKSAAP': f'CKSAAP_{SPLIT_TAG}_y_label&score{NUM_FOLDS}.csv',
    'OBC': f'OBC_{SPLIT_TAG}_y_label&score{NUM_FOLDS}.csv',
    'PSEAAC': f'PSEAAC_{SPLIT_TAG}_y_label&score{NUM_FOLDS}.csv',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def roc(labels_dict, scores_dict):
    font = {'family': 'arial',
            'size': 12}
    params = {
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 1.5
    }
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    pylab.rcParams['font.weight'] = 'bold'

    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot([0, 1], [0, 1], linewidth=1, color='grey', linestyle='--')

    colors = plt.cm.get_cmap('tab10', len(labels_dict))

    for i, (feature_name, y_true) in enumerate(labels_dict.items()):
        y_score = scores_dict[feature_name]
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr,
                 color=colors(i),
                 linewidth=1,
                 label=f'{feature_name} (AUC={auc:.4f})')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))

    plt.legend(
        prop={'size': 12},
        loc='lower right',
        frameon=False
    )
    plt.show()


def auc_11(fileplace, fold):
    feature_lable = {}
    feature_pscores = {}
    features = list(BASE_SCORE_FILES)
    reference = load_fixed_group_folds(
        FOLD_REFERENCE_PATH,
        n_splits=fold,
        seed=SEED,
    )
    keys = ['Sample_ID', 'Group_ID', 'Fold']
    merged = reference[keys + ['Label']].rename(columns={'Label': 'label'})

    for feature, filename in BASE_SCORE_FILES.items():
        path = f'{fileplace}/{filename}'
        df = pd.read_csv(path)
        required = {*keys, 'label', 'score'}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{filename} is missing E4 fold columns; rerun its grouped training script"
            )
        if df.duplicated(keys).any():
            raise ValueError(f"{filename} contains duplicate OOF samples")
        if len(df) != len(reference):
            raise ValueError(
                f"{filename} has {len(df)} rows; expected {len(reference)} E4-fold samples"
            )
        current = df[keys + ['label', 'score']].rename(
            columns={'label': f'{feature}_label', 'score': feature}
        )
        merged = merged.merge(current, on=keys, how='left', validate='one_to_one')
        if merged[feature].isna().any():
            raise ValueError(f"{filename} does not cover every E4-fold sample")
        if not merged[f'{feature}_label'].astype(int).eq(merged['label'].astype(int)).all():
            raise ValueError(f"{filename} labels do not match the E4 fold reference")
        merged = merged.drop(columns=[f'{feature}_label'])
        print(f"Loaded {feature}: {len(df)} aligned OOF predictions from {filename}")

    for feature in features:
        feature_pscores[feature] = merged[feature].values
        feature_lable[feature] = merged['label'].values
    return (
        merged[features].values,
        feature_pscores,
        feature_lable,
        merged['label'].values,
        merged['Sample_ID'].values,
        merged['Group_ID'].values,
        merged['Fold'].values,
    )


class DNN(nn.Module):
    def __init__(self, size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(size, 128)
        nn.init.ones_(self.fc1.bias)
        self.fc2 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(self.fc2(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc3(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc4(x))
        return x


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def training_DNN(pepID, groups, fold_assignments, x, y, feature_pscores, feature_lable, fold):

    feature_size = x.shape[1]

    y_label = []
    y_score = []
    peplist = []
    score_rows = []

    for count in range(1, fold + 1):
        set_seed(SEED + count)
        train_index = np.flatnonzero(fold_assignments != count)
        test_index = np.flatnonzero(fold_assignments == count)
        inner_train_index, val_index = group_train_validation_split(
            y[train_index],
            groups[train_index],
            validation_size=0.1,
            seed=SEED + count,
        )
        validation_index = train_index[val_index]
        train_index = train_index[inner_train_index]
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_val, y_val = x[validation_index], y[validation_index]

        train_dataset = CustomDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

        model = DNN(feature_size).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())

        best_auc_fold = 0
        best_model_state = None

        for epoch in range(50):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
                y_val_score = model(x_val_tensor).cpu().numpy()
            auc = roc_auc_score(y_val, y_val_score)
            print(f'Epoch {epoch + 1} - Validation AUC: {auc}')
            if auc > best_auc_fold:
                best_auc_fold = auc
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if best_model_state is None:
            raise RuntimeError(f"No valid meta-model selected for fold {count}")
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            best_pred_fold = model(x_test_tensor).cpu().numpy()

        y_label.append(list(y_test))
        y_score.append(list(best_pred_fold))
        peplist.append(list(pepID[test_index]))
        score_rows.extend(
            {
                'Sample_ID': pepID[index],
                'Group_ID': groups[index],
                'Fold': count,
                'label': int(y[index]),
                'score': float(score),
            }
            for index, score in zip(test_index, best_pred_fold.ravel())
        )

        model_path = (
            f'./trained_models/meta_model_{SPLIT_TAG}_fold_{count}_{fold}.pth'
        )
        torch.save(best_model_state, model_path)

    df_y = pd.DataFrame(score_rows)
    df_y.to_csv(
        f'./scores/meta_model_{SPLIT_TAG}_y_label&score{fold}.csv',
        index=False,
    )

    AUC_score = roc_auc_score(list(chain.from_iterable(y_label)), list(chain.from_iterable(y_score)))
    feature_lable['MetaModel_E4'] = list(chain.from_iterable(y_label))
    feature_pscores['MetaModel_E4'] = list(chain.from_iterable(y_score))

    print('———————————— final AUC：', AUC_score, ' ————————————')
    return AUC_score, feature_pscores, feature_lable

def main():
    set_seed(SEED)
    fold = NUM_FOLDS
    dataset, feature_pscores, feature_lable, labels, peplist, groups, fold_assignments = auc_11(
        './scores', fold
    )
    _, feature_pscores, feature_lable = training_DNN(
        peplist,
        groups,
        fold_assignments,
        dataset,
        labels,
        feature_pscores,
        feature_lable,
        fold,
    )
    roc(feature_lable, feature_pscores)


if __name__ == '__main__':
    main()
