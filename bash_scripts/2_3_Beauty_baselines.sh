cd ../src
python main_train_model.py category=Beauty \
    cf_weight=2.5 \
    confidence_type=loss_based \
    milestones=[10,90]

python main_train_model.py category=Beauty \
    cf_weight=2.5 \
    cf_loss_type=triplet \
    batch_size=128 \
    milestones=[10,90]
