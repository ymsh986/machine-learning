import os

import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import fbeta_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize, minimize_scalar

from imblearn.under_sampling import RandomUnderSampler

from lightgbm import LGBMModel


class LGBM:
    """LGBMModelのラッパー"""
    def __init__(self, Config, Util):
        self.model = None
        self.Config = Config
        self.Util = Util

    def build(self):
        self.model = LGBMModel(**self.Config.model_params)

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], early_stopping_rounds=self.Config.fit_params['early_stopping_rounds'], verbose=self.Config.fit_params['verbose'])
        #self.model.fit(tr_x, tr_y, eval_set=[(va_x, va_y)])

    def predict(self, x):
        preds = self.model.predict(x)
        return preds

    def save(self, filepath):
        # self.Util.dump(self.model, filepath)
        joblib.dump(self.model, filepath, compress=True)

    def load(self, filepath):
        # self.model = self.Util.load(filepath)
        self.model = joblib.load(filepath)

def skf(train, n_splits, random_state, target_col):
    """層化KFoldのインデックスのリストを作成"""
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    return list(skf.split(train, train[target_col]))


def opt_fbeta_threshold(y_true, y_pred):
    """fbeta score計算時のthresholdを最適化"""
    def opt_(x):
        return -fbeta_score(y_true, y_pred >= x, beta=7)
    result = minimize_scalar(opt_, bounds=(0, 1), method='bounded')
    print(result)
    best_threshold = result['x'].item()
    return best_threshold


def metrics(y_true, y_pred):
    """fbeta(beta=7)の閾値最適化評価関数"""
    bt = opt_fbeta_threshold(y_true, y_pred)
    print(f"bt:{bt}")
    score = fbeta_score(y_true, y_pred >= bt, beta=7)
    return score


def under_sampling(X, y, seed, strategy=None):

    if strategy is not None:
        sampler = RandomUnderSampler(random_state=seed, replacement=True, sampling_strategy=strategy)
    else:
        sampler = RandomUnderSampler(random_state=seed, replacement=True)

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    print(X.shape, " -> ", X_resampled.shape)
    print(y.shape, " -> ", y_resampled.shape)
    print("Major num / Minor num = ", (len(y_resampled) - y_resampled.sum()) / y_resampled.sum())
    return X_resampled, y_resampled


def train_cv_v2(X, y, Config, Util, cv, metrics, name, directory):
    oof = np.zeros(len(y))
    for i_fold, (tr_idx, va_idx) in enumerate(cv):
        filepath = os.path.join(directory, f"{name}_fold{i_fold+1}.pkl")
        tr_x, va_x = X.iloc[tr_idx].reset_index(drop=True), X.iloc[va_idx].reset_index(drop=True)
        tr_y, va_y = y.values[tr_idx], y.values[va_idx]

        ## UnderSampling
        count_positive = tr_y.sum()
        strategy = {0:count_positive * Config.ratio, 1:count_positive}
        tr_x, tr_y = under_sampling(tr_x, tr_y, Config.seeds[0], strategy)

        print(tr_x.shape)
        print(tr_y.shape)
        print(va_x.shape)
        print(va_y.shape)
        model = LGBM(Config, Util)
        model.build()
        model.fit(tr_x, tr_y, va_x, va_y)
        preds = model.predict(va_x)
        model.save(filepath)
        oof[va_idx] = preds

        score = metrics(np.array(va_y), np.array(preds))
        print(f"{name}_fold{i_fold+1} >>> val socre:{score:.4f}")

    score = metrics(np.array(y), oof)
    print(f"{name} >>> val score:{score:.4f}")
    return oof


def predict_cv_v2(X, Config, Util, name, directory):
    preds_fold = []
    for i_fold in range(Config.n_fold):
        filepath = os.path.join(directory, f"{name}_fold{i_fold+1}.pkl")
        print(f"{name}_fold{i_fold+1} inference")
        model = LGBM(Config, Util)
        model.build()
        model.load(filepath)
        preds = model.predict(X)

        preds_fold.append(preds)


    #preds = np.mean(preds_fold, axis=0)
    return preds


def tree_importance(X, y, Model, cv):
    """importance を取得"""
    feature_importance_df = pd.DataFrame()
    for i, (tr_idx, va_idx) in enumerate(cv):
        tr_x, va_x = X.values[tr_idx], X.values[va_idx]
        tr_y, va_y = y.values[tr_idx], y.values[va_idx]

        est = Model()
        est.build()

        est.fit(tr_x, tr_y, va_x, va_y)
        _df = pd.DataFrame()
        _df['feature_importance'] = est.model.feature_importances_
        _df['column'] = X.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column') \
                .sum()[['feature_importance']] \
                .sort_values('feature_importance', ascending=False).index[:50]
    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
    sns.boxenplot(data=feature_importance_df, y='column', x='feature_importance', order=order, ax=ax,
                  palette='viridis')
    fig.tight_layout()
    ax.grid()
    ax.set_title('feature importance')
    fig.tight_layout()
    return fig, feature_importance_df


def visualize_confusion_matrix(y_true, pred_label, height=.6, labels=None):
    """混合行列をプロット"""
    conf = confusion_matrix(y_true=y_true,
                            y_pred=pred_label,
                            normalize='true')

    n_labels = len(conf)
    size = n_labels * height
    fig, ax = plt.subplots(figsize=(size * 4, size * 3))
    sns.heatmap(conf, cmap='Blues', ax=ax, annot=True, fmt='.2f')
    ax.set_ylabel('Label')
    ax.set_xlabel('Predict')

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params('y', labelrotation=0)
        ax.tick_params('x', labelrotation=90)

    return fig
