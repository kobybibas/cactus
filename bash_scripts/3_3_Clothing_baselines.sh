cd ../src
python main_train_model.py category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=2.0 \
    confidence_type=loss_based

python main_train_model.py category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=128 \
    cf_weight=2.0 \
    cf_loss_type=triplet
