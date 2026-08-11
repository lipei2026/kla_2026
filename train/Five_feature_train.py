import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import roc_curve, roc_auc_score

from group_splits import group_train_validation_split, load_fixed_group_folds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def load_feature_with_e4_folds(feature_name, fold_reference):
    """Join deterministic feature rows to E4 folds by Sample_ID, never by row number."""
    feature_df = pd.read_excel(f'./six_features/{feature_name}.xlsx')
    required = {'UniProt_ID', 'Label'}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"{feature_name} is missing columns: {sorted(missing)}")
    if feature_df['UniProt_ID'].duplicated().any():
        raise ValueError(f"{feature_name} contains duplicate sample identifiers")

    feature_df = feature_df.rename(columns={'Label': 'Feature_Label'})
    reference = fold_reference[['Sample_ID', 'Group_ID', 'Fold', 'Label']]
    merged = reference.merge(
        feature_df,
        left_on='Sample_ID',
        right_on='UniProt_ID',
        how='left',
        validate='one_to_one',
    )
    if len(merged) != len(reference) or merged['Feature_Label'].isna().any():
        raise ValueError(f"{feature_name} could not be aligned to every resolved E4 sample")
    if not merged['Feature_Label'].astype(int).eq(merged['Label'].astype(int)).all():
        raise ValueError(f"{feature_name} labels do not match the E4 fold reference")
    return merged.drop(columns=['Feature_Label', 'UniProt_ID'])


def roc(labels_dict, scores_dict):
    font = {'family': 'arial',
            'weight': 'bold',
            'size': 20}
    params = {
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'lines.linewidth': 4
    }
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    pylab.rcParams['font.weight'] = 'bold'

    plt.figure(figsize=(7, 7), dpi=300)

    plt.plot([0, 1], [0, 1], linewidth=1, color='grey', linestyle='--', label='Random')

    colors = plt.cm.get_cmap('tab10', len(labels_dict))

    for i, (feature_name, y_true) in enumerate(labels_dict.items()):
        y_score = scores_dict[feature_name]
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr,
                 color=colors(i),
                 linewidth=3,
                 label=f'{feature_name} (AUC={auc:.3f})')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))

    plt.legend(
        prop={'size': 12},
        loc='lower right',
        frameon=False
    )
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.show()


def get_col(fn):
    dist = {
        'ACF': 510,
        'AAINDEX': 510,
        'CKSAAP': 441,
        'OBC': 1122,
        'PSEAAC': 20
    }
    i = 1
    columns = []
    while i <= dist[fn]:
        columns.append(str(i))
        i += 1
    return columns, dist[fn]


def prep_dataset(feature_name):
    global dataset
    col, size = get_col(feature_name)
    x = dataset[col]
    y = dataset['Label']
    sample_ids = dataset['Sample_ID']
    groups = dataset['Group_ID']
    folds = dataset['Fold']
    return x, y, sample_ids, groups, folds, size


hidden_config = {
    'ACF': {'hidden': [1024, 512], 'dropout': 0.3, 'bn': True},
    'AAINDEX': {'hidden': [512, 256], 'dropout': 0.4, 'bn': True},
    'OBC': {'hidden': [2048, 512], 'dropout': 0.5, 'bn': True},
    'CKSAAP': {'hidden': [768, 256], 'dropout': 0.4, 'bn': True},
    'PSEAAC': {'hidden': [64, 32], 'dropout': 0.2, 'bn': False}
}


class DNN(nn.Module):
    def __init__(self, size, hidden_sizes, dropout_rate, use_bn):
        super(DNN, self).__init__()
        layers = []
        input_size = size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def training_DNN(feature_name, fold):
    print('\n')
    print('————————————' + feature_name + '————————————')
    os.makedirs(f'./trained_models/{feature_name}', exist_ok=True)

    x, y, pepID, groups, fold_assignments, feature_size = prep_dataset(feature_name)
    x = x.values
    y = y.values
    pepID = pepID.values
    groups = groups.values
    fold_assignments = fold_assignments.values

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
        config = hidden_config.get(feature_name)
        hidden_sizes = config['hidden']
        dropout_rate = config['dropout']
        use_bn = config['bn']
        model = DNN(feature_size, hidden_sizes, dropout_rate, use_bn).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())
        best_auc = 0
        best_model_state = None
        for epoch in range(100):
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
            auc_score = roc_auc_score(y_val, y_val_score)
            print(f' Fold {count} {feature_name} , {epoch + 1}  AUC :{auc_score}')

            if auc_score > best_auc:
                best_auc = auc_score
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if best_model_state is None:
            raise RuntimeError(f"No valid model selected for {feature_name} fold {count}")
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            best_prediction = model(x_test_tensor).cpu().numpy()

        y_label.append(list(y_test))
        y_score.append(list(best_prediction))
        peplist.append(list(pepID[test_index]))
        score_rows.extend(
            {
                'Sample_ID': pepID[index],
                'Group_ID': groups[index],
                'Fold': count,
                'label': int(y[index]),
                'score': float(score),
            }
            for index, score in zip(test_index, best_prediction.ravel())
        )

        if best_model_state is not None:
            torch.save(best_model_state,
                       r'./trained_models/' + feature_name + r'/' + feature_name +
                       f'{count}_{fold}_{SPLIT_TAG}_DNN_.pth')

    from itertools import chain
    df_y = pd.DataFrame(score_rows)
    df_y.to_csv(
        f'./scores/{feature_name}_{SPLIT_TAG}_y_label&score{fold}.csv',
        index=False,
    )

    AUC_score = roc_auc_score(list(chain.from_iterable(y_label)), list(chain.from_iterable(y_score)))
    print('————————————', feature_name, 'final AUC:', AUC_score, '————————————')
    return AUC_score, list(chain.from_iterable(y_label)), list(chain.from_iterable(y_score))


features = ['ACF', 'AAINDEX', 'CKSAAP', 'OBC', 'PSEAAC']
feature_scores = []
feature_lable = {}
feature_pscores = {}
set_seed(SEED)
fold_reference = load_fixed_group_folds(
    FOLD_REFERENCE_PATH,
    n_splits=NUM_FOLDS,
    seed=SEED,
)
for feature in features:
    fold = NUM_FOLDS
    dataset = load_feature_with_e4_folds(feature, fold_reference)
    score, y_label, y_score = training_DNN(feature, fold)
    feature_scores.append(score)
    feature_lable[feature] = y_label
    feature_pscores[feature] = y_score

roc(feature_lable, feature_pscores)
