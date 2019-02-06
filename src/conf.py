"""Основные параметры."""
import logging

# Конфигурация логгера
logging.basicConfig(level=logging.INFO)

# Пути к данным
DATA_RAW = "../raw/train.csv"
DATA_SUB = "../raw/sample_submission.csv"
DATA_TEST = "../raw/test/{}.csv"
DATA_PROCESSED = "../processed/"

# Параметры признаков
BLOCKS = 30
LEARNING_RATE = 0.03
DEPTH = 10
