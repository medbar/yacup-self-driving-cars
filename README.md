# Решение занявшее 13-е место в Self Driving Cars на YaCup24. 
## Окружение
Все необходимое, чтобы установить окружение есть в setup.sh
Задачи запускаются с помощью [inex-launcher](https://github.com/speechpro/inex-launcher) тулы, которая инициализирует и выполняет задачу по ее OmegaConf конфигу. 
## Подготовка данных
Надо скачать ./YandexCup2024v2, далее или просто запустить `python prepare_data.py`, туда я скопировал нужный код подготовки данных из ipynb. Или, если надо разобраться как я к этому коду пришел, то можно пройти по ячейкам в normalization.ipynb, quant-v2.ipynb и prepare_for_training-2.ipynb

## Финальный сабмит (скор 2.03): 
Фичи использовались с шагом 20мс, но перед входом в rnn/transformer резались на патчи с окном 10 фреймов (200мс). 

* lstm модель, обученная по 245 входным кадрам предсказывать следующие 750.
* Работает как авторегрессионная модель, но после 245 входа локация маскируется.
* Обучалась на MSE, в конце несколько эпох на L1.
* Конфиг модели можно найти в exp/local_v2/train_lstm_like5_3.3_alldata_l1/final_config.yaml
  
Чтобы заинференсить ее, запустите `bash MAKE_SUBMIT.sh`. **Warning**: по умолчанию инферится на cuda. Если cuda нет, то make_submit_2.03.yaml:140 надо указать cpu 
Predict step устроен в Autoregressive манере, но на самом деле модели это не нужно. Можно предсказать все 750 отсчетов за раз.
## Дополнительная модель (скор 2.04 во фьюжене с train_lstm_like5_3.3_alldata_l1):
* Transformer aed модель, которая по 245 входным кадрам с известной локализацией (вход энкодера) предсказывает 750 локализаций для 750 фичей контроля (вход декодера). 
* exp/local_v2/train_aed_6_4_alldata_l1/final_config.yaml
Предсказывает все 750 отсчетов за раз, из-за чего работает быстро. 

## Обучение 
обучение lstm 
```
bash train_args.sh local_v2/train_lstm_like5_3.2_alldata.yaml
bash train_args.sh local_v2/train_lstm_like5_3.3_alldata.yaml
bash train_args.sh local_v2/train_lstm_like5_3.3_alldata_l1.yaml
```
обучение aed 
```
bash train_args.sh local_v2/train_aed_6_4_alldata.yaml
bash train_args.sh local_v2/train_aed_6_4_alldata_l1_ptest.yaml
```


 
