import os

import logging
import datetime

import joblib

class Config:
    name_v1 = "lgb"

    model_params = dict(n_estimators=10000,
                        num_leaves=31,
                        objective="binary",
                        learning_rate=0.01,
                        colsample_bytree=0.3,
                        class_weight="balanced",
                        importance_type="gain")

    fit_params = dict(early_stopping_rounds=100,
                      verbose=100)

    n_fold = 3
    seeds = [777]
    target_col = "judgement"
    submit = False
    debug = False
    ratio = 2


class Logger:
    """log を残す用のクラス"""
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, 'Experiment.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class Util:
    """pkl保存&load"""
    @classmethod
    def dump(cls, value, path):
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
