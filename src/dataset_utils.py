import logging
import os
import os.path as osp
import pickle
import time
import uuid
from io import BytesIO
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


class TransformBatch:
    def __init__(
        self,
        img_transforms: Callable[[Image.Image], torch.Tensor] = None,
        is_load_default_transform: bool = True,
        is_train: bool = False,
        img_transform_resize: int = 256,
        img_transform_crop: int = 224,
        is_torch_stack: bool = True,
    ):

        self.is_torch_stack = is_torch_stack

        if is_load_default_transform is True:
            if is_train is True:
                img_transforms = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            img_transform_crop
                        ),  # Default value 224 is aligend with ImageNet https://pytorch.org/vision/stable/models.html
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                        ),
                        transforms.RandomGrayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),  # Normalize is aligned with ImageNet training
                    ]
                )
            else:
                img_transforms = transforms.Compose(
                    [
                        transforms.Resize(
                            img_transform_resize
                        ),  # Default value 256 is aligend with ImageNet
                        transforms.CenterCrop(
                            img_transform_crop
                        ),  # Default value 224 is aligend with ImageNet
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),  # Normalize is aligned with ImageNet training
                    ]
                )
        self._img_transforms = img_transforms

    def __call__(self, batch):
        """Interprets bytes and replaces them with its interpretation"""
        batch_imgs, batch_labels, batch_cf_vectors, batch_is_labeled = [], [], [], []
        for i in range(len(batch["image"])):
            raw_img = batch["image"][i]

            # It is a tuple of (all labels, True)
            label = batch["label"][0][i]
            is_labeled = batch["is_labeled"][0][i]

            # It is a list of tuples [(cf_vector, True)]
            cf_vector = batch["cf_vector"][i][0]

            try:
                byte = BytesIO(raw_img.numpy().view())
                byte.seek(0)
                img = Image.open(byte, mode="r").convert("RGB")

            except Exception as e:
                if False:  # Used for debug
                    logger.info(
                        f'Fail load image. {batch["image_path"][i]=} {type(byte)=} {e=} {byte=}'
                    )
                continue

            if self._img_transforms:
                img = self._img_transforms(img)

            batch_imgs.append(img)
            batch_labels.append(label)
            batch_is_labeled.append(is_labeled)
            batch_cf_vectors.append(cf_vector)

        if self.is_torch_stack is True:
            batch_imgs = torch.stack(batch_imgs)
            batch_labels = torch.stack(batch_labels)
            batch_is_labeled = torch.stack(batch_is_labeled)
            batch_cf_vectors = torch.stack(batch_cf_vectors)

        return batch_imgs, batch_labels, batch_is_labeled, batch_cf_vectors


def load_dataset_pkl(category: str, pkl_path: str, local_dir_path: str = None):
    if local_dir_path is not None:
        # Read meta data from local dir if exists
        local_pkl_path = osp.join(local_dir_path, osp.basename(pkl_path))
        if not osp.exists(local_pkl_path):
            logger.info(f"Downloading localy: {local_pkl_path=} {pkl_path=}")
            pkl_dict = read_data_from_manifold(pkl_path)
            with open(local_pkl_path, "wb") as f:
                pickle.dump(pkl_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Load pkl dict from local dir
        pkl_path = local_pkl_path

    with open(local_pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)
    return pkl_dict


def load_cf_vectors(
    base_dir: str,
    embed_suffix: str = "embed_items.pt",
    bias_suffix: str = "embed_items_bias.pt",
    is_use_cf_bias: bool = True,
) -> torch.Tensor:
    item_cf_vector_path = osp.join(base_dir, embed_suffix)
    item_cf_vector_bias_path = osp.join(base_dir, bias_suffix)

    embed_items = read_data_from_manifold(
        item_cf_vector_path,
        is_from_pkl=True,
    )
    embed_items_bias = read_data_from_manifold(
        item_cf_vector_bias_path,
        is_from_pkl=True,
    )

    if is_use_cf_bias is True:
        embed_items = torch.cat((embed_items, embed_items_bias), dim=-1)
    return embed_items


def create_on_box_dataloader(
    dataset, num_workers: int, dpp_server_num_worker_threads: int, phase: str = ""
) -> OnBoxDataLoader:

    os.putenv("GLOG_minloglevel", "3")  # reduce logs
    data_loading_options = DataLoadingOptions(
        dpp_server_glog_options=GlogOptions(disable_info_logs=True),
        dpp_server_num_worker_threads=dpp_server_num_worker_threads,
        num_python_transform_workers=num_workers,
        pin_memory=True,
    )

    dataloder = OnBoxDataLoader(
        identity=f"Loader {phase} {uuid.uuid4()}",
        session=PristineSession(dataset=dataset),
        data_loading_options=data_loading_options,
        rank=0,
        discovery_method=SingleInstanceDiscoveryMethod(),  # For runs that are not in FBLearber
    )
    return dataloder


def get_datasets(
    category: str,
    data_dir: str,
    cf_vector_base_dir: str,
    is_use_cf_bias: bool = True,
    labeled_ratio: float = 1.0,
    batch_size: int = 1,
    train_set_repeat: int = 1,
    is_shuffle_train: bool = True,
    num_workers: int = 1,
    train_set_ratio: float = 0.8,
    local_dir_path: str = None,
    train_batch_transform=None,
    test_batch_transform=None,
):
    if local_dir_path is not None:
        os.makedirs(local_dir_path, exist_ok=True)

    t0 = time.time()
    dataset_pkl_path = osp.join(data_dir, f"{category}.pkl")
    pkl_dict = load_dataset_pkl(category, dataset_pkl_path, local_dir_path)
    meta_df = pkl_dict["meta_df"]
    logger.info(f"Finish load_dataset_pkl in {time.time() -t0 :.2f} {len(meta_df)=}")

    t0 = time.time()
    if cf_vector_base_dir is not None:
        cf_vectors = load_cf_vectors(cf_vector_base_dir, is_use_cf_bias=is_use_cf_bias)
    else:
        cf_vectors = torch.zeros(len(meta_df), 32)
    cf_vector_dim = cf_vectors.size(-1)
    logger.info(
        f"Finish load embed_items in {time.time() - t0:.2f} sec. {cf_vectors.shape=}"
    )

    # Add cf vector to meta df
    meta_df["cf_vector"] = cf_vectors.tolist()

    # Add manifold path to img name
    meta_df["manifold_img_path"] = meta_df.img_name.apply(
        lambda x: osp.join(data_dir, category, x)
    )

    # Construct dataset
    meta_df_train = meta_df.sample(frac=train_set_ratio)
    meta_df_train["is_labeled"] = torch.rand(len(meta_df_train)) > 1 - labeled_ratio

    meta_df_test = meta_df.drop(meta_df_train.index)
    meta_df_test["is_labeled"] = True

    train_set_data = [
        (
            row["manifold_img_path"],
            row["categories"],
            row["cf_vector"],
            row["is_labeled"],
        )
        for i, row in meta_df_train.iterrows()
    ]

    test_set_data = [
        (
            row["manifold_img_path"],
            row["categories"],
            row["cf_vector"],
            row["is_labeled"],
        )
        for i, row in meta_df_test.iterrows()
    ]

    # Degine the dataset template
    column_names = [
        "image_path",
        "label",
        "cf_vector",
        "is_labeled",
    ]
    schema = ["image_path", "image", "label", "cf_vector", "is_labeled"]
    enrichments = [
        ManifoldEnrichment(
            lookup_value="image_path",
            output_column="image",
            options=ManifoldEnrichment.Options(num_retries=4, timeout_msec=10000),
        )
    ]

    # Construct dataloaders
    if train_batch_transform is None:
        train_batch_transform = TransformBatch(is_train=False)
    train_dataset = (
        InlineDataset(column_names=column_names, data=train_set_data)
        .enrichments(enrichments)
        .schema(schema)
        .batch(batch_size)
        .transform(train_batch_transform)
    )

    if train_set_repeat > 1:
        train_dataset.repeat(
            train_set_repeat - 1
        )  # minus 1 since number of passes is 1 + train_set_repeat

    if test_batch_transform is None:
        test_batch_transform = TransformBatch(is_train=False)
    test_dataset = (
        InlineDataset(column_names=column_names, data=test_set_data)
        .enrichments(enrichments)
        .schema(schema)
        .batch(batch_size)
        .use_num_splits(1)
        .transform(test_batch_transform)
    )

    dataset_meta = {
        "train_set_size": len(train_set_data),
        "test_set_size": len(test_set_data),
        "classes": list(pkl_dict["cate_map"].keys()),
        "cf_vector_dim": cf_vector_dim,
    }
    return train_dataset, test_dataset, dataset_meta
