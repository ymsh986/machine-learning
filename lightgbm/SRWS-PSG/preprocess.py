import os

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import torch

import texthero as hero
import transformers


class AutoSequenceVectorizer:

    def __init__(self, model_name='bert-base-uncased', max_len=128):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.max_len = max_len

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        output = self.model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = output['last_hidden_state'], output['pooler_output']

        if torch.cuda.is_available():
            # 0番目は [CLS] token, 768 dim の文章特徴量
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()


def basic_text_features_transform(input_df, text_columns, cleansing_hero=None, name=""):
    """basic な text 特徴量"""
    def _get_features(dataframe, column):
        _df = pd.DataFrame()
        _df[column + name + '_num_chars'] = dataframe[column].apply(len)
        _df[column + name + '_num_exclamation_marks'] = dataframe[column].apply(lambda x: x.count('!'))
        _df[column + name + '_num_question_marks'] = dataframe[column].apply(lambda x: x.count('?'))
        _df[column + name + '_num_punctuation'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '.,;:'))
        _df[column + name + '_num_symbols'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '*&$%'))
        _df[column + name + '_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        _df[column + name + '_num_unique_words'] = dataframe[column].apply(lambda x: len(set(w for w in x.split())))
        _df[column + name + '_words_vs_unique'] = _df[column + name + '_num_unique_words'] / _df[column + name + '_num_words']
        _df[column + name + '_words_vs_chars'] = _df[column + name + '_num_words'] / _df[column + name + '_num_chars']
        return _df

    output_df_ = pd.DataFrame()
    output_df_[text_columns] = input_df[text_columns].fillna('missing').astype(str)
    output_lst = []
    for c in text_columns:
        if cleansing_hero is not None:
            output_df_[c] = cleansing_hero(output_df_, c)
        output_df = _get_features(output_df_, c)
        output_lst.append(output_df)
    output_df = pd.concat(output_lst, axis=1)
    return output_df


def vectorize_text(input_df,
                   text_columns,
                   cleansing_hero=None,
                   vectorizer=CountVectorizer(),
                   transformer=TruncatedSVD(n_components=128),
                   name='html_count_svd'):
    """countベースのtext特徴量"""

    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].fillna('missing').astype(str)
    features = []
    for c in text_columns:
        if cleansing_hero is not None:
            output_df[c] = cleansing_hero(output_df, c)

        sentence = vectorizer.fit_transform(output_df[c])
        feature = transformer.fit_transform(sentence)
        num_p = feature.shape[1]
        feature = pd.DataFrame(feature, columns=[f'{c}_' + name + str(num_p) + f'={i:03}' for i in range(num_p)])
        features.append(feature)
    output_df = pd.concat(features, axis=1)
    return output_df


def cleansing_hero_only_text(input_df, text_col):
    # get text only
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,  # すべてのテキストを小文字に
        hero.preprocessing.remove_digits,  # Remove all blocks of digits.
        hero.preprocessing.remove_punctuation,  # すべてのstring.punctuationを削除します（！ "＃$％＆ '（）* +、-。/：; <=>？@ [\] ^ _` {|}〜）
        hero.preprocessing.remove_diacritics,  # 文字列からすべてのアクセントを削除
        hero.preprocessing.remove_stopwords,  # ストップワードとは一般的で役に立たない等の理由で処理対象外とする単語のこと
        hero.preprocessing.remove_whitespace,  # 単語間の空白をすべて削除
        hero.preprocessing.stem  # ステミングとは動詞など複数の言い回しが存在する品詞の原型を取り出すこと
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts


def get_basic_text_features(input_df):
    output_df = basic_text_features_transform(input_df,
                                              text_columns=["title", "abstract"],
                                              cleansing_hero=cleansing_hero_only_text)
    return output_df


def get_tfidf_features__svd64(input_df):
    output_df = vectorize_text(input_df,
                               text_columns=["title", "abstract"],
                               cleansing_hero=cleansing_hero_only_text,
                               vectorizer=TfidfVectorizer(min_df=0.001, max_df=0.99),
                               transformer=TruncatedSVD(n_components=64),
                               name="tfidf_svd_")
    return output_df


def get_count_features__svd64(input_df):
    output_df = vectorize_text(input_df,
                               text_columns=["title", "abstract"],
                               cleansing_hero=cleansing_hero_only_text,
                               vectorizer=CountVectorizer(min_df=0.001, max_df=0.99),
                               transformer=TruncatedSVD(n_components=64),
                               name="count_svd_")
    return output_df


def get_scibert_features__pca128(input_df):
    """scibertで特徴抽出"""
    vectorizer = AutoSequenceVectorizer(model_name="allenai/scibert_scivocab_uncased",
                                        max_len=256)
    texts = input_df["title"] + " " + input_df["abstract"].fillna("")
    text_vecs = np.array([vectorizer.vectorize(x) for x in tqdm(texts)])
    pca = PCA(n_components=128)
    text_vecs = pca.fit_transform(text_vecs)

    output_df = pd.DataFrame(text_vecs, columns=[f"scibert_pca_vecs={i:03}" for i in range(text_vecs.shape[1])])
    return output_df


def preprocess(train, test):
    """前処理の実行関数"""
    input_df = pd.concat([train, test]).reset_index(drop=True)  # reset_indexでindexを振り直す. drop=Trueで元のindexを削除する
    funcs = [get_basic_text_features,
             get_scibert_features__pca128,
             get_tfidf_features__svd64,
             get_count_features__svd64]

    output = []
    for func in funcs:
        filepath = os.path.join(EXP_FEATS_PATH, f"{func.__name__}.pkl")
        if os.path.isfile(filepath):
            _df = Util.load(filepath)
        else:
            _df = func(input_df)
            Util.dump(_df, filepath)  # 作った特徴量は保存
        output.append(_df)
    output = pd.concat(output, axis=1)

    train_x = output.iloc[:len(train)]
    train_y = train[Config.target_col]
    test_x = output.iloc[len(train):].reset_index(drop=True)

    return train_x, train_y, test_x


class Renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])
