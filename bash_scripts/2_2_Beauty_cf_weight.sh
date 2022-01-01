cd ../src
python main_train_model.py -m category=Beauty \
    cf_weight=0.0,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5 \
    milestones=[10,90]