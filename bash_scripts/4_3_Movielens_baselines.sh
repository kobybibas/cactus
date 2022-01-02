cd ../src
python main_train_model.py category=movielens \
    arch=mobbilenet \
    batch_size=128 \
    confidence_type=loss_based

python main_train_model.py category=movielens \
    arch=mobbilenet \
    batch_size=64 \
    cf_loss_type=triplet
