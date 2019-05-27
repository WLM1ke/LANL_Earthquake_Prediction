# LANL Earthquake Prediction

Решение для конкурса [Kaggle](https://www.kaggle.com/c/LANL-Earthquake-Prediction):

Около 600 млн. строк данных о сейсмической активности - необходимо предсказать оставшееся время до землятресения.

- LANL_conv_revert_pir
    - ver1 - 1.979 (1.565) +++

- LANL_conv_revert_pir_less_conv
    - ver3 - 64 - 1.969 (1.581)
    
- LSTM_last
    - ver2 - 128, 2 - 1.952 (1.575)
    
- Big_conv
    - ver1


Нормированный dot.prod как мера похожести волн - для атокодировщика
https://arxiv.org/pdf/1804.06812.pdf
https://www.kaggle.com/hsinwenchang/mfcc-randomforestregressor-catboostregressor?scriptVersionId=13214884
https://arxiv.org/pdf/1801.04503.pdf
https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/90791
https://www.kaggle.com/tarunpaparaju/lanl-earthquake-prediction-new-features
https://arxiv.org/pdf/1708.03888.pdf

https://www.kaggle.com/kashnitsky/how-to-download-submission-file-without-committing
https://www.tensorflow.org/api_docs/python/tf/signal/frame
