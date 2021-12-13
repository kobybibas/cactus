

## Train recommender
Avilable categories: Clothing_Shoes_and_Jewelry/Toys_and_Games/Home_and_Kitchen
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
python train_recommender.py category=Clothing_Shoes_and_Jewelry num_samples_threshold=12
python process_labels.py category=Clothing_Shoes_and_Jewelry
python train_model.py -m category=Clothing_Shoes_and_Jewelry  \
    cf_weight=0.0,2.0 \
    labeled_ratio=0.4,0.6,0.8,0.9,1.0
```

```
python train_recommender.py category=Toys_and_Games
python process_labels.py category=Toys_and_Games
python train_model.py -m category=Toys_and_Games  \
    cf_weight=0.0,2.0 \ 
    labeled_ratio=0.4,0.6,0.8,0.9,1.0
```

```
python train_recommender.py category=Home_and_Kitchen
python process_labels.py category=Home_and_Kitchen
python train_model.py -m category=Home_and_Kitchen  \ 
    cf_weight=0.0,2.0 \ 
    labeled_ratio=0.4,0.6,0.8,0.9,1.0
```

```
python train_recommender.py category=Beauty
python process_labels.py category=Beauty
python train_model.py -m category=Beauty lr=0.01 \
    cf_weight=0.0,2.0 \
    labeled_ratio=0.4,0.6,0.8,0.9,1.0


```

## Ablation
```
python train_model.py -m category=Beauty lr=0.01 cf_weight=2.0 labeled_ratio=1.0 cf_topk_loss_ratio=0.9
```

