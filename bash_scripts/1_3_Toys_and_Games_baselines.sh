cd ../src
python main_train_model.py category=Toys_and_Games \
    cf_weight=2.5 \
    confidence_type=loss_based

python main_train_model.py category=Toys_and_Games \
    cf_weight=2.5 \
    cf_loss_type=triplet \
    batch_size=128
