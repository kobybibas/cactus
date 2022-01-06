cd ../src
python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=128 \
    milestones=[10,80,95] \
    confidence_type=loss_based

python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=64 \
    milestones=[10,80,95] \
    cf_loss_type=triplet
