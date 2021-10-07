import logging
import os
import os.path as osp
import time

import hydra
import pandas as pd
from omegaconf import DictConfig
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.datasets.folder import default_loader as img_loader


logger = logging.getLogger(__name__)


def merge_label_hierarchy(label_list: list, category: str) -> list:
    # label_list: list of list.

    label_processed_list = []
    for label in label_list:

        # Keep only labels with that belong to the top level category
        if label[0] == category:
            # Remove labels with one letter
            label = [label_i for label_i in label if len(label_i) > 1]

            # Join hierarchy
            label_processed_list.append(" + ".join(label))

    return label_processed_list


def remove_downlevel_hierarchy(labels: list, label_mapper: dict) -> list:
    outs = []
    for label in labels:
        if label in label_mapper.keys():
            outs.append(label_mapper[label])
        else:
            outs.append(label)
    return outs


@hydra.main(
    config_path="../configs",
    config_name="process_labels",
)
def process_labels(cfg: DictConfig):
    out_dir = os.getcwd()
    logger.info(cfg)
    logger.info(os.getcwd())

    # Load df
    t0 = time.time()
    meta_path = osp.join(cfg.data_dir, f"meta_{cfg.category}.pkl")
    meta_df = pd.read_pickle(meta_path)
    logger.info(f"Loadded meta_df in {time.time() -t0:.2f} sec. {len(meta_df)=}")

    t0 = time.time()
    review_path = osp.join(cfg.data_dir, f"reviews_{cfg.category}.pkl")
    reivew_df = pd.read_pickle(review_path)
    logger.info(f"Loadded review_df in {time.time() -t0:.2f} sec. {len(reivew_df)=}")

    # Keep only items with reviews
    t0 = time.time()
    asin = reivew_df.drop_duplicates(subset=["asin"])["asin"].tolist()
    meta_df = meta_df[meta_df["asin"].isin(asin)]
    logger.info(f"Item with reviews {time.time() -t0:.2f} sec. {len(meta_df)=}")

    # Add image paths
    meta_df["img_path"] = meta_df["imUrl"].apply(
        lambda x: osp.join(cfg.data_dir, cfg.category, osp.basename(str(x)))
    )

    # Keep only items with images
    img_exists = []
    for img_path in meta_df["img_path"]:
        if osp.exists(img_path):
            try:
                img_loader(img_path)
                img_exists.append(True)
            except:
                img_exists.append(False)
        else:
            img_exists.append(False)
    meta_df["img_exists"] = img_exists
    logger.info(f"Img exsists {meta_df.img_exists.sum()}/{len(meta_df)}")
    df = meta_df[meta_df["img_exists"] == True][["asin", "img_path", "categories"]]

    # Merge label hierarchy:
    # For example: [Clothing, Shoes & Jewelry + Girls + Clothing + Swim] -> [Clothing, Shoes & Jewelry + Girls + Clothing]
    df["merged_labels"] = df["categories"].apply(
        lambda x: merge_label_hierarchy(x, category=cfg.toplevel_label)
    )

    # Count number of samples for each category: remove downlevel category if there are not enough samples
    min_num_samples = 0
    n_iter = 0
    while min_num_samples <= cfg.num_samples_threshold:
        label_count = pd.value_counts(
            list(chain.from_iterable(df["merged_labels"].tolist()))
        )
        min_num_samples = label_count.min()

        logger.info(
            f"[{n_iter}] {len(label_count)=} {(label_count<cfg.num_samples_threshold).sum()=}"
        )
        logger.info(f"\n{label_count[label_count<cfg.num_samples_threshold]}")

        label_mapper = {
            label: " + ".join(label.split(" + ")[:-1])
            for label in label_count[label_count <= cfg.num_samples_threshold].index
        }

        df["merged_labels"] = df["merged_labels"].apply(
            lambda labels: remove_downlevel_hierarchy(labels, label_mapper=label_mapper)
        )

        n_iter += 1

    # Save category

    # Encode to Multilabel vector
    mlb = MultiLabelBinarizer()
    df["label_vec"] = mlb.fit_transform(df["merged_labels"].tolist()).tolist()
    logger.info(f"\n{df.head()}")

    # Save results
    out_path = osp.join(out_dir, "label_count.csv")
    label_count.to_csv(out_path, header=False)

    out_path = osp.join(out_dir, "df_w_labels.pkl")
    df = df.reset_index()
    df.to_pickle(out_path)
    logger.info(f"Save to {out_path}")

    out_path = osp.join(out_dir, "label_mapper.csv")
    pd.DataFrame(mlb.classes_).to_csv(out_path, header=False)
    logger.info(f"Save to {out_path}")

    logger.info("Finish")


if __name__ == "__main__":
    process_labels()
