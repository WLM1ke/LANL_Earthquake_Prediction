"""Основные параметры."""
import logging

# Конфигурация логгера
logging.basicConfig(level=logging.INFO)

# Пути к данным
DATA_RAW = "../raw/train.csv"
DAT_SUB = "../raw/sample_submission.csv"
DATA_TEST = "../raw/test/{}.csv"
DATA_PROCESSED = "../processed/"
