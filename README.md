

## Train recommender
Avilable categories: Clothing_Shoes_and_Jewelry Toys_and_Games
```
python train_recommender.py category=Clothing_Shoes_and_Jewelry
```

## Process labels
```
python process_labels.py category=Clothing_Shoes_and_Jewelry
```

## Train vision model 
```
python train_model.py -m category=Clothing_Shoes_and_Jewelry cf_weight=0.0,2.0 labeled_ratio=0.4,0.6,0.8,0.9,1.0
```


# All
```
python train_recommender.py category=Toys_and_Games
python process_labels.py category=Toys_and_Games toplevel_label="Toys & Games"
python train_model.py category=Toys_and_Games  \
    cf_weight=2.0 \
    labeled_ratio=0.9 \
    cf_vector_df_path=../outputs/Toys_and_Games/train_recommender_20211206_085539/cf_df.pkl \
    train_df_path=../outputs/Toys_and_Games/process_label_20211206_091606/df_train.pkl \
    test_df_path=../outputs/Toys_and_Games/process_label_20211206_091606/df_test.pkl
python train_model.py -m category=Toys_and_Games  \ 
    cf_weight=0.0,2.0 \ 
    labeled_ratio=0.4,0.6,0.8,0.9,1.0 \
    cf_vector_df_path=../outputs/Toys_and_Games/train_recommender_20211206_085539/cf_df.pkl \
    train_df_path=../outputs/Toys_and_Games/process_label_20211206_091606/df_train.pkl \
    test_df_path=../outputs/Toys_and_Games/process_label_20211206_091606/df_test.pkl
``


