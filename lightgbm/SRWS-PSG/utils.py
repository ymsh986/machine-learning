import os

import logging
import datetime

import joblib


class Config:
    name_v1 = "lgb"

    model_fixed_params = dict(n_estimators=10000,
                              objective="binary",
                              colsample_bytree=0.7,
                              importance_type="gain")

    """
    model_variable_params = dict(num_leaves=[15, 31, 63],
                                 max_depth=[6, 8, 10],
                                 min_gain_to_split=[0, 0.1])
    model_variable_params = dict(num_leaves=[31, 63],
                                 max_depth=[8, 10],
                                 learning_rate=[0.01, 0.001])
    """
    model_variable_params = dict(num_leaves=[31],
                                 max_depth=[10],
                                 learning_rate=[0.01])

    fit_params = dict(early_stopping_rounds=100,
                      verbose=100)

    n_fold = 5
    seeds = [777]
    target_col = "judgement"
    submit = False
    debug = False
    ratio = 20


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
