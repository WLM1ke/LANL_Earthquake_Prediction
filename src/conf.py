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
CROP_Y = 20

# Параметры Catboost
LEARNING_RATE = 0.1
DEPTH = 6
