cd ../src
python main_train_recommender.py category=Beauty test_size=0.0
python main_process_labels.py category=Beauty
python main_train_model.py -m category=Beauty \
    lr=0.1 \
    cf_weight=0.0,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5 \
    labeled_ratio=1.0