from typing import List

import numpy as np
from cornac.data import Reader
from cornac.data.reader import read_text
from cornac.metrics.ranking import MeasureAtK
from cornac.utils import cache
import os.path as osp
import pandas as pd


class HitRate(MeasureAtK):
    def __init__(self, k=-1):
        super().__init__(name="HitRate@{}".format(k), k=k)

    def compute(self, gt_pos, pd_rank, **kwargs):
        tp, _, _ = MeasureAtK.compute(self, gt_pos, pd_rank, **kwargs)
        return tp


class AmazonClothing:
    def __init__(
        self, data_dir: str, category: str = "Clothing_Shoes_and_Jewelry"
    ) -> None:
        self.data_dir = data_dir
        self.category = category
        self.review_path = osp.join(self.data_dir, f"reviews_{category}.pkl")
        self.rating_path = osp.join(self.data_dir, f"rating_{category}.txt")
        if not osp.exists(self.rating_path):
            self.convert_reaview_pkl_to_rating()

    def convert_reaview_pkl_to_rating(self):
        review_df = pd.read_pickle(
            osp.join(self.data_dir, f"reviews_{self.category}.pkl")
        )

        # Algin to rating.txt format
        review_df = review_df[["reviewerID", "asin", "overall"]]

        review_df.to_csv(self.rating_path, sep="\t", index=False, header=False)

    def load_feedback(self, reader: Reader = None) -> List:
        # Load the user-item ratings, scale: [1,5]
        reader = Reader(bin_threshold=1.0) if reader is None else reader
        return reader.read(self.rating_path, sep="\t")

    # def load_feedback(self, reader: Reader = None) -> List:
    #     """Load the user-item ratings, scale: [1,5]

    #     Parameters
    #     ----------
    #     reader: `obj:cornac.data.Reader`, default: None
    #         Reader object used to read the data.

    #     Returns
    #     -------
    #     data: array-like
    #         Data in the form of a list of tuples (user, item, rating).
    #     """
    #     fpath = cache(
    #         url="https://static.preferred.ai/cornac/datasets/amazon_clothing/rating.zip",
    #         unzip=True,
    #         relative_path="amazon_clothing/rating.txt",
    #         cache_dir=self.data_dir,
    #     )
    #     reader = Reader(bin_threshold=1.0) if reader is None else reader
    #     return reader.read(fpath, sep="\t")
