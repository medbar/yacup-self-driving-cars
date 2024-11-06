#!/bin/bash - 

#best model
inex -s . --log-level INFO \
    ./make_submit_2.03.yaml

ls exp/local_v2/train_lstm_like5_3.3_alldata_l1/submits/*.csv.gz
