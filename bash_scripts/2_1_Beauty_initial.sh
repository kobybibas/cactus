cd ../src
python main_train_recommender.py category=Beauty test_size=0.0
python main_process_labels.py category=Beauty
python main_train_model_cf_based.py category=Beauty