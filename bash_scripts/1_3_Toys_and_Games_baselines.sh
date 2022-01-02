cd ../src
python main_train_model.py category=Toys_and_Games \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=2.5 \
    confidence_type=loss_based

python main_train_model.py category=Toys_and_Games \
    arch=resnet18 \
    batch_size=128 \
    cf_weight=2.5 \
    cf_loss_type=triplet
