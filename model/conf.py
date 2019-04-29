"""Основные параметры."""
import logging

# Конфигурация логгера
from sklearn import model_selection

logging.basicConfig(level=logging.INFO)

# Пути к данным
DATA_RAW = "../raw/train.csv"
DATA_SUB = "../raw/sample_submission.csv"
DATA_TEST = "../raw/test/{}.csv"
DATA_PROCESSED = "../processed/"

SEED = 284702
FOLDS = 8
K_FOLDS = model_selection.KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
