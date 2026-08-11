"""Five-dimensional MetaModel ablation using the exact E4 protein folds.

Only the input features and output tag differ from Meta_model.py. Network
architecture, optimizer, epochs, batch size, validation protocol, and seeds
are inherited unchanged so the 5D and 7D experiments are directly comparable.
"""

import Meta_model as meta


EXPERIMENT_TAG = 'E4folds_grouped_5D'
SELECTED_SCORE_FILES = {
    'E4': f'ESM2_E4_center_k_mean_grouped_y_label&score{meta.NUM_FOLDS}.csv',
    'LSTM': f'LSTM_{meta.SPLIT_TAG}_y_label&score{meta.NUM_FOLDS}.csv',
    'AAINDEX': f'AAINDEX_{meta.SPLIT_TAG}_y_label&score{meta.NUM_FOLDS}.csv',
    'CKSAAP': f'CKSAAP_{meta.SPLIT_TAG}_y_label&score{meta.NUM_FOLDS}.csv',
    'OBC': f'OBC_{meta.SPLIT_TAG}_y_label&score{meta.NUM_FOLDS}.csv',
}


def main():
    # Meta_model functions read these globals at call time. Replacing only
    # these two values keeps every other training setting identical to 7D.
    meta.BASE_SCORE_FILES = SELECTED_SCORE_FILES
    meta.SPLIT_TAG = EXPERIMENT_TAG
    meta.set_seed(meta.SEED)

    fold = meta.NUM_FOLDS
    dataset, feature_scores, feature_labels, labels, sample_ids, groups, folds = meta.auc_11(
        './scores', fold
    )
    if dataset.shape[1] != 5:
        raise RuntimeError(f'Expected five MetaModel inputs, found {dataset.shape[1]}')
    print(f'Running 5D MetaModel ablation with inputs: {list(SELECTED_SCORE_FILES)}')

    _, feature_scores, feature_labels = meta.training_DNN(
        sample_ids,
        groups,
        folds,
        dataset,
        labels,
        feature_scores,
        feature_labels,
        fold,
    )
    meta.roc(feature_labels, feature_scores)


if __name__ == '__main__':
    main()
