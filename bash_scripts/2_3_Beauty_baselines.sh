cd ../src
python main_train_model.py category=Beauty \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=1.0 \
    confidence_type=loss_based \
    milestones=[10,90] \
    conf_max_min_ratio=3.0

python main_train_model.py category=Beauty \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=1.0 \
    confidence_type=num_intercations \
    milestones=[10,90] \
    conf_max_min_ratio=3.0

python main_train_model.py category=Beauty \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=1.0 \
    confidence_type=pos_label_loss_based \
    milestones=[10,90] \
    conf_max_min_ratio=3.0

python main_train_model.py category=Beauty \
    arch=resnet18 \
    batch_size=128 \
    cf_weight=1.0 \
    cf_loss_type=triplet \
    milestones=[10,90]
