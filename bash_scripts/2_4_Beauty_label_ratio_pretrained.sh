cd ../src
python main_train_model.py -m category=Beauty \
    cf_weight=2.5 \
    labeled_ratio=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    milestones=[10,90]

python main_train_model.py -m category=Beauty \
    cf_weight=0.0 \
    labeled_ratio=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    milestones=[10,90]