import logging
import os
import os.path as osp
import pickle
import random
import time

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from data_utils import DataHelper
from tqdm import tqdm

logger = logging.getLogger(__name__)

def remove_infrequent_items(df: pd.DataFrame, min_counts: int = 5) -> pd.DataFrame:
    counts = df["asin"].value_counts()
    df = df[df["asin"].isin(counts[counts >= min_counts].index)]
    logger.info("items with < {} interactoins are removed".format(min_counts))
    return df


def remove_infrequent_users(df: pd.DataFrame, min_counts: int = 10) -> pd.DataFrame:
    counts = df["reviewerID"].value_counts()
    df = df[df["reviewerID"].isin(counts[counts >= min_counts].index)]
    logger.info("users with < {} interactoins are removed".format(min_counts))
    return df


def select_sessions(df: pd.DataFrame, mins, maxs) -> pd.DataFrame:
    selected_id = []
    for reviewerID, group in tqdm(df.groupby("reviewerID")):
        time_len = len(group["unixReviewTime"].unique())
        if time_len >= mins and time_len <= maxs:
            selected_id.append(reviewerID)

    df = df[df["reviewerID"].isin(selected_id)]
    logger.info(
        "selected session({0} <= session <= {1}):{2}".format(mins, maxs, len(df))
    )
    return df


def select_meta(df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    items = df["asin"].unique()
    return meta_df[meta_df["asin"].isin(items)]


def remove_item_from_review_df(
    review_df: pd.DataFrame, meta_df: pd.DataFrame
) -> pd.DataFrame:
    items = meta_df["asin"].unique()
    return review_df[review_df["asin"].isin(items)]


def build_map(df: pd.DataFrame, col_name: str):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name + "_org"] = df[col_name]
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def build_set_for_recommendation(
    reviews_df: pd.DataFrame,
    example_count: int,
    last_review_for_test_ratio: float = 0.1,
):

    test_pos_list, train_pos_list = [], []
    for _, reviewer_df in tqdm(reviews_df.groupby("reviewerID")):
        reviewer_df.sort_values(by="unixReviewTime", inplace=True)

        # Take the last 10% items as positive
        num_item = len(reviewer_df)
        num_test_items = int(last_review_for_test_ratio * num_item)
        if num_test_items > 0:
            test_pos_list.append(reviewer_df.tail(num_test_items))

        # Take all other to train
        num_train_items = num_item - num_test_items
        train_pos_list.append(reviewer_df.head(num_train_items))

    train_df = pd.concat(train_pos_list)
    test_df = pd.concat(test_pos_list)

    train_set = train_df[["reviewerID", "asin"]].to_numpy().astype("int32")
    test_set = test_df[["reviewerID", "asin"]].to_numpy().astype("int32")

    assert len(test_set) + len(train_set) == example_count
    return train_set, test_set


def remove_infrequent_labels(
    meta_df: pd.DataFrame, num_sample_per_class_threshold: int = 20
) -> pd.DataFrame:
    logger.info(
        "labels with < {} samples are removed".format(num_sample_per_class_threshold)
    )
    # Remove items with label that have small number of samples
    label_int, label_str = pd.factorize(meta_df["categories"])
    num_samples_per_class = np.bincount(label_int)
    label_to_keep = num_samples_per_class > num_sample_per_class_threshold
    logger.info(f"Num of label. [Org New]=[{len(label_to_keep)} {label_to_keep.sum()}]")

    meta_df_filt = meta_df[np.in1d(label_int, np.where(label_to_keep == True)[0])]
    logger.info(f"Num items [Org New]=[{len(meta_df)} {len(meta_df_filt)}]")
    return meta_df_filt


@hydra.main(
    config_path="../configs",
    config_name="remap_id",
)
def remap_id(cfg: DictConfig):
    t_start = time.time()
    logger.info(cfg)
    logger.info(f"{os.getcwd()=}")

    random.seed(1234)

    manifold_path = cfg.data_dir
    for category in cfg.category_list:
        t0 = time.time()

        review_path = osp.join(manifold_path, f"reviews_{category}.pkl")
        meta_path = osp.join(manifold_path, f"meta_{category}.pkl")

        # Load dataframes
        data_downloader = DataHelper(
            is_debug=cfg.is_debug,
            is_override=cfg.is_override,
        )
        reviews_df = data_downloader.read_pickle(review_path)
        reviews_df["unixReviewTime"] = reviews_df["unixReviewTime"] // 3600 // 24

        meta_df = data_downloader.read_pickle(meta_path)
        meta_df = meta_df[["asin", "categories", "imUrl"]]
        meta_df["categories"] = meta_df["categories"].map(lambda x: x[-1][-1])
        meta_df["img_name"] = (
            meta_df["imUrl"].astype("str").apply(lambda x: osp.basename(x))
        )
        meta_df_org = meta_df.copy()

        # Filter reviews
        # reviews_df = remove_infrequent_users(reviews_df, 10)
        # reviews_df = remove_infrequent_items(reviews_df, 8)
        # reviews_df = select_sessions(reviews_df, 4, 90) # TODO: verify we dont need it
        meta_df = select_meta(reviews_df, meta_df)
        logger.info(
            "num of users:{}, num of items:{}".format(
                len(reviews_df["reviewerID"].unique()), len(reviews_df["asin"].unique())
            )
        )

        # Remove items that belong to label with small number of samples
        # meta_df = remove_infrequent_labels(meta_df, cfg.num_sample_per_class_threshold)
        # reviews_df = remove_item_from_review_df(reviews_df, meta_df)

        # Map
        asin_map, asin_key = build_map(meta_df, "asin")
        cate_map, cate_key = build_map(meta_df, "categories")
        revi_map, revi_key = build_map(reviews_df, "reviewerID")

        user_count, item_count, cate_count, example_count = (
            len(revi_map),
            len(asin_map),
            len(cate_map),
            reviews_df.shape[0],
        )
        logger.info(f"{user_count=} {item_count=} {cate_count=} {example_count=}")

        meta_df = meta_df.sort_values("asin")
        meta_df = meta_df.reset_index(drop=True)
        reviews_df["asin_org"] = reviews_df["asin"]
        reviews_df["asin"] = reviews_df["asin"].map(lambda x: asin_map[x])
        reviews_df = reviews_df.sort_values(["reviewerID", "unixReviewTime"])
        reviews_df = reviews_df.reset_index(drop=True)

        item_cate_list = [meta_df["categories"][i] for i in range(len(asin_map))]
        item_cate_list = np.array(item_cate_list, dtype=np.int32)

        # Train-test split
        train_set, test_set = build_set_for_recommendation(
            reviews_df,
            example_count,
            cfg.last_review_for_test_ratio,
        )
        logger.info(f"{len(train_set)=} {len(test_set)=}")

        # Save results
        out_dict = {
            "reviews_df": reviews_df,
            "meta_df": meta_df,
            "meta_df_org": meta_df_org,  
            "item_cate_list": item_cate_list,
            "cate_map": cate_map,
            "user_count": user_count,
            "item_count": item_count,
            "cate_count": cate_count,
            "example_count": example_count,
            "train_set": train_set,
            "test_set": test_set,
        }

        out_pkl = osp.join(cfg.data_dir, category + ".pkl")
        with open(out_pkl, 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Finish {out_pkl=} in {time.time() -t0 :.2f} sec")

    logger.info(f"Finish remap_id in {time.time() -t_start :.2f} sec")


if __name__ == "__main__":
    remap_id()
