cd ../src
python main_train_model.py -m category=Beauty \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=1.0 \
    labeled_ratio=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    is_pretrained=false \
    epochs=300 \
    milestones=[150,250,295]

python main_train_model.py -m category=Beauty \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=0.0 \
    labeled_ratio=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    is_pretrained=false \
    epochs=300 \
    milestones=[150,250,295]