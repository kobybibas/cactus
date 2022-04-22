# Collaborative Image Understanding


## Download data
Pinterest: https://nms.kcl.ac.uk/netsys/datasets/social-curation/dataset.html

MovieLens: https://www.kaggle.com/ghrzarea/movielens-20m-posters-for-machine-learning

Amazon product data (Clothing/Toys): http://jmcauley.ucsd.edu/data/amazon/

## Run experiments
To replicate the experiments for the Pinterst dataset

```
cd bash_script

# Train recommender
./2_1_pinterest_initial.sh

# Optimize for the best cf weight
./2_2_Pinterest_cf_weight.sh

# Train baselines
./2_3_Pinterest_baselines.sh

# Train with different label ratio
./2_4_Pinterest_label_ratio

```

Follow the other files in the bash_script folder for the other datasets.

Then update result file
```
/home/ubuntu/cactus/outputs/pinterest/results.yaml
```
and run 
```
python main_predict_testset.py dataset_name=pinterest
```
Lastly to produce the figures run
```
python main_evaluate_methods.py 
```