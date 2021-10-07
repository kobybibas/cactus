import logging
import os.path as osp
import time
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split
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
        cf_vector = row["embs"]
        target = row["label_vec"]
        is_labeled = row["is_labeled"]

        cf_vector = torch.tensor(cf_vector)
        target = torch.tensor(target)

        # Load image
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, cf_vector, target, is_labeled


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
    label_df_path: str,
    cf_vector_df_path: str,
    labeled_ratio: float = 1.0,
    train_set_ratio: float = 0.8,
):

    t0 = time.time()
    label_df = pd.read_pickle(label_df_path)
    cf_df = pd.read_pickle(cf_vector_df_path)
    logger.info(f"Loaded df in {time.time() -t0 :.2f}")

    # Join dfs
    t0 = time.time()
    df = pd.merge(label_df, cf_df, on=["asin"], how="inner")
    logger.info(
        f"merge df in {time.time() -t0 :.2f}. {len(df)=} {len(label_df)=} {len(cf_df)=}"
    )

    # Construct dataset
    n_iter = 0
    min_label_count = 0
    while min_label_count < 2:
        df_train, df_test = train_test_split(df, train_size=train_set_ratio)

        test_labels = np.array(df_test.label_vec.to_list())
        train_labels = np.array(df_test.label_vec.to_list())
        min_label_count = min(
            train_labels.sum(axis=0).min(), test_labels.sum(axis=0).min()
        )
        logger.info(f"[{n_iter}] train-test split {min_label_count=}")
        n_iter += 1

    df_train["is_labeled"] = torch.rand(len(df_train)) > 1 - labeled_ratio
    df_test["is_labeled"] = True

    train_dataset = DfDatasetWithCF(df_train, transform=train_transform)
    test_dataset = DfDatasetWithCF(df_test, transform=test_transform)

    num_classes = len(df["label_vec"].iloc[0])
    cf_vector_dim = len(df["embs"].iloc[0])

    dataset_meta = {
        "train_set_size": len(df_train),
        "test_set_size": len(df_test),
        "num_classes": num_classes,
        "cf_vector_dim": cf_vector_dim,
    }

    return train_dataset, test_dataset, dataset_meta
