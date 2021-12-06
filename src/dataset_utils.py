import logging
import os.path as osp
import time
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import numpy as np

logger = logging.getLogger(__name__)


class DfDatasetWithCF(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, target_transform=None) -> None:
        super().__init__()
        self.df = df
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["img_path"]
        pos_img_path = row["pos_img_path"] if "pos_img_path" in row else row["img_path"]
        cf_vector = row["embs"]
        target = row["label_vec"]
        is_labeled = row["is_labeled"]

        cf_vector = torch.tensor(cf_vector)
        target = torch.tensor(target)

        # Load image
        image = self.loader(img_path)
        image_pos = self.loader(pos_img_path)

        if self.transform is not None:
            image = self.transform(image)
            image_pos = self.transform(image_pos)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, image_pos, cf_vector, target, is_labeled


# Transformation as ImageNet training
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize,
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)


def get_datasets(
    df_train_path: str,
    df_test_path: str,
    cf_vector_df_path: str,
    labeled_ratio: float = 1.0,
):

    t0 = time.time()
    df_train = pd.read_pickle(df_train_path)
    df_test = pd.read_pickle(df_test_path)
    cf_df = pd.read_pickle(cf_vector_df_path)
    logger.info(
        f"Loaded df in {time.time() -t0 :.2f} sec. {len(df_train)=} {len(df_test)=} {len(cf_df)=}"
    )

    # Add CF vectors
    t0 = time.time()
    df_train = pd.merge(df_train, cf_df, on=["asin"], how="inner")
    df_test = pd.merge(df_test, cf_df, on=["asin"], how="inner")
    logger.info(
        f"merge df in {time.time() -t0 :.2f} sec. {len(df_train)=} {len(df_test)=} {len(cf_df)=}"
    )

    # Add positive CF
    if "pos_asin" in df_train.columns:
        pos_img_path = pd.merge(
            df_train[["asin", "pos_asin"]],
            pd.concat((df_train[["asin", "img_path"]], df_test[["asin", "img_path"]])),
            left_on=["pos_asin"],
            right_on=["asin"],
            how="left",
        )["img_path"]
        df_train["pos_img_path"] = pos_img_path
        pos_img_path = pd.merge(
            df_test[["asin", "pos_asin"]],
            pd.concat((df_train[["asin", "img_path"]], df_test[["asin", "img_path"]])),
            left_on=["pos_asin"],
            right_on=["asin"],
            how="left",
        )["img_path"]
        df_test["pos_img_path"] = pos_img_path

    # Hide labels
    df_train["is_labeled"] = torch.rand(len(df_train)) > 1.0 - labeled_ratio
    df_test["is_labeled"] = True

    # Define positive weight: Since positives are much less than negatives, increase their weights
    train_labels = np.array(
        df_train[df_train["is_labeled"] == True].label_vec.to_list()
    )
    pos_weight = len(train_labels) / (train_labels.sum(axis=0) + 1e-6)

    # Construct dataset
    train_dataset = DfDatasetWithCF(df_train, transform=train_transform)
    test_dataset = DfDatasetWithCF(df_test, transform=test_transform)

    # Get metadata
    num_classes = len(df_train["label_vec"].iloc[0])
    cf_vector_dim = len(df_train["embs"].iloc[0])

    dataset_meta = {
        "train_set_size": len(df_train),
        "test_set_size": len(df_test),
        "num_classes": num_classes,
        "cf_vector_dim": cf_vector_dim,
    }

    return train_dataset, test_dataset, dataset_meta, pos_weight
