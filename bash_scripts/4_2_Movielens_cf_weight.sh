cd ../src
python main_train_model.py -m category=movielens \
    arch=mobbilenet \
    batch_size=128 \
    cf_weight=0.0,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5