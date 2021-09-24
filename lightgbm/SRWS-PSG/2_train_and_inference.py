import os
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import fbeta_score

from utils import Config, Util
from functions import skf, opt_fbeta_threshold, metrics
from functions import single_train, predict, visualize_confusion_matrix
from functions import create_grid_parameter_list

import argparse

parser = argparse.ArgumentParser(description='for SRWS-PSG.')
parser.add_argument('--exp-name', help='experiment name', required=True)
args = parser.parse_args()

ROOT_DIR = './results'
EXP_NAME = args.exp_name
DATASETS_DIRNAME = "./datasets"
EXP_DIR = os.path.join(ROOT_DIR, EXP_NAME)
EXP_MODEL_PATH = os.path.join(EXP_DIR, "model")
EXP_PREDS_PATH = os.path.join(EXP_DIR, "preds")
EXP_FIG_PATH = os.path.join(EXP_DIR, "fig")
EXP_FEATS_PATH = os.path.join(EXP_DIR, "features")
EXP_SUBMISSION_PATH = os.path.join(EXP_DIR, "submission")

AGGREGATION_REUSLTS_DIR = os.path.join(ROOT_DIR, "aggregation_results")


def create_experiment_directories(EXP_NAME="EXP"):

    # 実験ごとのフォルダを作成
    dirs = [EXP_DIR, EXP_SUBMISSION_PATH, EXP_MODEL_PATH,
            EXP_PREDS_PATH, EXP_FIG_PATH, EXP_FEATS_PATH,
            AGGREGATION_REUSLTS_DIR]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_datasets():

    train = pd.read_csv(f"{DATASETS_DIRNAME}/train.csv")
    test = pd.read_csv(f"{DATASETS_DIRNAME}/test.csv")
    train_x = pd.read_csv(f"{DATASETS_DIRNAME}/train_x.csv", index_col=0)
    train_y = pd.read_csv(f"{DATASETS_DIRNAME}/train_y.csv", index_col=0)
    test_x = pd.read_csv(f"{DATASETS_DIRNAME}/test_x.csv", index_col=0)
    print("train_x shape : ", train_x.shape)
    print("train_y shape : ", train_y.shape)
    print("test_x shape : ", test_x.shape)
    return train, test, train_x, train_y, test_x


def remake_datasets(train_x, test_x):

    unnecessary_num_col_list = ['title_num_exclamation_marks',
                                'title_num_question_marks',
                                'title_num_punctuation',
                                'title_num_symbols',
                                'abstract_num_exclamation_marks',
                                'abstract_num_question_marks',
                                'abstract_num_punctuation',
                                'abstract_num_symbols']

    unnecessary_vs_col_list = ['title_words_vs_unique',
                               'title_words_vs_chars',
                               'abstract_words_vs_unique',
                               'abstract_words_vs_chars']

    train_x = train_x.drop(unnecessary_num_col_list, axis=1)
    test_x = test_x.drop(unnecessary_num_col_list, axis=1)
    train_x = train_x.drop(unnecessary_vs_col_list, axis=1)
    test_x = test_x.drop(unnecessary_vs_col_list, axis=1)

    # 暫定処理（csv内の列名に重複があるとエラーになるので、重複を解消する）
    # unnecessary_CountVectorizer_col_list = [f'{c}_count_svd_64={i:03}' \
    #                                       　for c in ["title", "abstract"] for i in range(64)]
    scibert_col_list = [f'scibert_pca_vecs={i:03}' for i in range(128)]
    CountVectorizer_col_list = [f'count_svd_64={i:03}' for i in range(64)] \
                                + [f'count_svd_64={i:03}.1' for i in range(64)]
    TfidfVectorizer_col_list = [f'tfidf_svd_64={i:03}' for i in range(64)] \
                                + [f'tfidf_svd_64={i:03}.1' for i in range(64)]

    train_x_datasets = {}
    """
    train_x_datasets['all'] = train_x
    train_x_datasets['scibert_only'] = train_x.drop(CountVectorizer_col_list + TfidfVectorizer_col_list, axis=1)
    train_x_datasets['count_only'] = train_x.drop(scibert_col_list + TfidfVectorizer_col_list, axis=1)
    train_x_datasets['tfidf_only'] = train_x.drop(CountVectorizer_col_list + scibert_col_list, axis=1)
    train_x_datasets['scibert_count'] = train_x.drop(TfidfVectorizer_col_list, axis=1)
    train_x_datasets['count_tfidf'] = train_x.drop(scibert_col_list, axis=1)
    """
    train_x_datasets['tfidf_scibert'] = train_x.drop(CountVectorizer_col_list, axis=1)
    test_x_datasets = {}
    """
    test_x_datasets['all'] = test_x
    test_x_datasets['scibert_only'] = test_x.drop(CountVectorizer_col_list + TfidfVectorizer_col_list, axis=1)
    test_x_datasets['count_only'] = test_x.drop(scibert_col_list + TfidfVectorizer_col_list, axis=1)
    test_x_datasets['tfidf_only'] = test_x.drop(CountVectorizer_col_list + scibert_col_list, axis=1)
    test_x_datasets['scibert_count'] = test_x.drop(TfidfVectorizer_col_list, axis=1)
    test_x_datasets['count_tfidf'] = test_x.drop(scibert_col_list, axis=1)
    """
    test_x_datasets['tfidf_scibert'] = test_x.drop(CountVectorizer_col_list, axis=1)
    return train_x_datasets, test_x_datasets


def execute_training(Config, train, X, y, dataset_key='default', grid_param_list=None):

    print("# ============= # Training # ============= #")
    max_score = 0
    best_param = {}
    max_bt = 0
    max_idx = 0
    for idx, grid_param in enumerate(grid_param_list):
        print(f"parameter pattern {idx+1}/{len(grid_param_list)}")
        oof_df = pd.DataFrame()
        for seed in Config.seeds:
            param = dict(**Config.model_fixed_params, **grid_param)
            name = f"{Config.name_v1}-{dataset_key}-{seed}-grid{idx}"
            oof = single_train(X=X, y=y, Config=Config, Util=Util,
                               cv=skf(train, n_splits=Config.n_fold,
                                      random_state=seed,
                                      target_col=Config.target_col),
                               metrics=metrics,
                               name=name,
                               directory=EXP_MODEL_PATH,
                               model_params=param)

            oof_df[name] = oof
        oof_df
        oof_df.to_csv(os.path.join(EXP_PREDS_PATH, "oof.csv"), index=False)

        # get oof score & best threshold
        y_true = train[Config.target_col]
        y_pred = oof_df.mean(axis=1)

        best_threshold = opt_fbeta_threshold(y_true.values, y_pred.values)
        oof_score = fbeta_score(y_true, y_pred >= best_threshold, beta=7)
        comments = f"score:{oof_score:.4f}/threshold:{best_threshold}"

        print(f"------- {name} result -------")
        print(f"parameters {param}")
        print(comments)

        if oof_score > max_score:
            max_score = oof_score
            max_bt = best_threshold
            best_param = param
            max_idx
            print(" -> best score update!!!")
        print("-----------------------")

    fig = visualize_confusion_matrix(y_true, y_pred >= best_threshold)
    fig.savefig(os.path.join(EXP_FIG_PATH, f"cm-{name}.png"), dpi=300)
    return max_score, max_bt, best_param, max_idx


def execute_inference(Config, Util, X, param_num, best_param, dataset_key):

    print("# ============= # Inference # ============= #")
    preds_df = pd.DataFrame()
    for seed in Config.seeds:
        name = f"{Config.name_v1}-{dataset_key}-{seed}-grid{param_num}"
        preds = predict(X, Config, Util, name, EXP_MODEL_PATH, best_param)
        preds_df[name] = preds

    return preds_df


def binarize_results(preds, best_threshold):

    return (preds.values >= best_threshold) * 1


def majority_vote(df):

    return [*map(lambda x: np.argmax(np.bincount(x)), df.values)]


def make_submission_data(pred):

    sub_df = pd.read_csv(os.path.join(f"{DATASETS_DIRNAME}/sample_submit.csv"),
                         header=None)

    sub_df.columns = ["id", "judgement"]
    sub_df["judgement"] = pred

    filepath = os.path.join(EXP_SUBMISSION_PATH, f"{EXP_NAME}.csv")
    sub_df.to_csv(filepath, index=False, header=False)


def save_experiment_data_to_csv():

    filepath = os.path.join(AGGREGATION_REUSLTS_DIR, "results.csv")

    # openしてからファイルの存在を確かめると必ず存在することになるので事前に確認
    if os.path.isfile(filepath):
        file_exists = True
    else:
        file_exists = False

    with open(filepath, mode='a', encoding='utf-8') as f:
        if file_exists is False:
            f.write('Time,EXP_NAME,Score,TP,FP,FN,TN,bt,n_fold,ratio\n')
        f.write(f'0,{EXP_NAME},0,0,0,0,0,0,0,0\n')


def main():

    # setup experiment directories
    create_experiment_directories(EXP_NAME=EXP_NAME)

    # load dataset
    train, test, train_x, train_y, test_x = load_datasets()

    # preprocess
    train_x_datasets, test_x_datasets = remake_datasets(train_x, test_x)

    grid_param_list = create_grid_parameter_list(Config.model_variable_params)

    # experiment each datasets
    best_score = {}
    best_threshold = {}
    best_param = {}
    best_param_num = {}
    preds_df = pd.DataFrame()

    for i, dataset_key in enumerate(train_x_datasets):

        print(f"dataset pattern {i+1}/{len(train_x_datasets)}")
        # start training
        best_score[dataset_key], best_threshold[dataset_key], best_param[dataset_key], best_param_num[dataset_key] = execute_training(Config, train, train_x_datasets[dataset_key], train_y, dataset_key, grid_param_list)

        # start inference
        preds_df[dataset_key] = execute_inference(Config, Util, test_x_datasets[dataset_key], best_param_num[dataset_key], best_param[dataset_key], dataset_key)

        # binarize inference results by best_threshold
        preds_df[dataset_key] = binarize_results(preds_df[dataset_key], best_threshold[dataset_key])

    # save results
    preds_df.to_csv(os.path.join(EXP_PREDS_PATH, "preds.csv"), index=False)

    for dataset_key in preds_df.keys():

        print("******************************************************")
        print(dataset_key)
        print(f"best_score : {best_score[dataset_key]}")
        print(f"best_threshold : {best_threshold[dataset_key]}")
        print(f"best_param : {best_param[dataset_key]}")
        if best_score[dataset_key] < 0.83:
            print(f"{dataset_key} is low score ... drop")
            preds_df = preds_df.drop(dataset_key, axis=1)
    print("******************************************************")
    print(preds_df.shape)

    # choice majority result
    majority_pred = majority_vote(preds_df)

    # make data for submit
    make_submission_data(majority_pred)
    sys.exit(0)

    # save experiment data
    save_experiment_data_to_csv()


if __name__ == "__main__":
    main()
