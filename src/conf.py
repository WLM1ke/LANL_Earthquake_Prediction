"""Основные параметры."""
import logging

# Конфигурация логгера
logging.basicConfig(level=logging.INFO)

# Пути к данным
DATA_RAW = "../raw/train.csv"
DATA_SUB = "../raw/sample_submission.csv"
DATA_TEST = "../raw/test/{}.csv"
DATA_PROCESSED = "../processed/"

# Параметры генерации признаков
REBUILD = False
PASSES = 1  # 40
WEIGHTED = False
GROUP_WEIGHTS = list([0.146, 0.134, 0.146, 0.134, 0.131, 0.029, 0.111])
GROUP_WEIGHTS.append(1 - sum(GROUP_WEIGHTS))

# Параметры Catboost
LEARNING_RATE = 0.1
DEPTH = 6
