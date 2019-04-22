# LANL Earthquake Prediction

Решение для конкурса [Kaggle](https://www.kaggle.com/c/LANL-Earthquake-Prediction):

Около 600 млн. строк данных о сейсмической активности - необходимо предсказать оставшееся время до землятресения.

- Автокодировщик над fft abs с учетом тестового семпла
- Первый коментарий https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/84601#latest-517009
- https://www.kaggle.com/taqanori/trying-mfcc-mel-frequency-cepstral-coefficients
- https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89
- Даются два сегмента - находятся ли они на расстоянии менее 150000
- https://www.kaggle.com/mrganger/lanl-finding-l-estimators-via-pca
- https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694

https://sites.psu.edu/chasbolton/2018/09/12/continuous-acoustic-data/


- LANL_conv_revert_pir
    - ver1 - 1.979 (1.565) +++

- LANL_conv_revert_pir_less_conv
    - ver3 - 64 - 1.969 (1.581)
    
- LSTM_last
    - ver2 - 128, 2 - 1.952 (1.575)
    
- Big_conv
    - ver1
    


Лучшие модели фичей не использовали. В основном использовал сжатый через maxpool и очищенный от тренда через diff временной ряд.

Модели:
1. Keras bilstm + attention (на основе публичного, лучшая соло модель)
4. Keras resnet1d
6. Bossvs (слабая соло, но сильная в ансамбле)

    
https://www.kaggle.com/abhishek/quite-a-few-features-1-51
