cd ../src
python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=128 \
    lr=0.01 \
    milestones=[40] \
    epochs=45 \
    cf_weight=5.0 \
    confidence_type=loss_based \
    cf_loss_type=exp

python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=96 \
    lr=0.01 \
    milestones=[40] \
    epochs=45 \
    cf_weight=2.5 \
    confidence_type=pos_label_loss_based \
    cf_loss_type=exp \
    conf_max_min_ratio=3.0

python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=128 \
    lr=0.01 \
    milestones=[40] \
    epochs=45 \
    cf_weight=2.5 \
    confidence_type=pos_label_loss_based_norm \
    cf_loss_type=exp \
    conf_max_min_ratio=3.0

python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=128 \
    lr=0.01 \
    milestones=[40] \
    epochs=45 \
    cf_weight=2.5 \
    confidence_type=num_intercations \
    cf_loss_type=exp

python main_train_model.py category=movielens \
    arch=mobilenet \
    batch_size=128 \
    lr=0.01 \
    milestones=[40] \
    epochs=45 \
    cf_weight=2.5 \
    cf_loss_type=triplet


