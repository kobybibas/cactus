cd ../src
python main_train_model.py -m category=movielens \
    arch=mobilenet \
    batch_size=128 \
    cf_weight=2.0 \
    milestones=[10,80,95] \
    labeled_ratio=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

python main_train_model.py -m category=movielens \
    arch=mobilenet \
    batch_size=128 \
    cf_weight=0.0 \
    milestones=[10,80,95] \
    labeled_ratio=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0