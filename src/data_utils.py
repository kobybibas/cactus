import gzip
import io
import logging
import os
import os.path as osp
from pickle import dump
from typing import List

import pandas as pd
import requests
from src.manifold_utils import (
    create_manifold_directory,
    file_exists_on_manifold,
    save_data_to_manifold,
    read_data_from_manifold,
    list_all_files,
)
from tqdm import tqdm


logger = logging.getLogger(__name__)


class DataHelper:
    def __init__(self, is_manifold: bool, is_debug: bool, is_override: bool):
        self.is_manifold = is_manifold
        self.is_debug = is_debug
        self.is_override = is_override
        self.save_data_to_manifold = save_data_to_manifold

    def is_img_path(self, path: str) -> bool:
        if path.lower().endswith((".jpg", ".png", ".jepg", ".gif", ".tiff")):
            return True
        else:
            return False

    def is_exist(self, path: str):
        if self.is_manifold:
            is_path_exist = file_exists_on_manifold(path)
        else:
            is_path_exist = osp.exists(path)

        if self.is_override:
            is_path_exist = False
        return is_path_exist

    def download_url(
        self,
        url: str,
        dst: str = None,
        is_force_download: bool = False,
    ):
        if self.is_debug:
            logger.info(f"download_url: {url=} {dst=} {is_force_download=}")

        if dst is None:
            dst = os.path.basename(url)
        if is_force_download is False and self.is_exist(dst):
            return

        r = requests.get(url, proxies={"http": "fwdproxy:8080"})
        if self.is_manifold:
            save_data_to_manifold(dst, r.content, is_pkl=not self.is_img_path(dst))
        else:
            with open(dst, "wb") as f:
                f.write(r.content)

    def ungzip_file(self, path_src: str, path_dst: str):
        if self.is_debug:
            logger.info(f"ungzip_file: {path_src=} {path_dst=}")

        if self.is_exist(path_dst):
            return

        elif self.is_manifold:
            gzip_data = read_data_from_manifold(path_src)
            data = gzip.decompress(gzip_data)
            save_data_to_manifold(path_dst, data)
        else:
            with gzip.open(path_src, "rb") as f:
                gzip_content = f.read()

            with open(path_dst, "wb") as f:
                f.write(gzip_content)

    def read_pickle(self, pkl_path: str) -> pd.DataFrame:
        if self.is_debug:
            logger.info(f"pd.read_pickle {pkl_path}")

        if self.is_manifold:
            pkl_data = read_data_from_manifold(pkl_path)
            df = pd.read_pickle(pkl_data)
        else:
            df = pd.read_pickle(pkl_path)
        return df

    def save_df_as_pkl(self, json_path: str, pkl_path: str):
        if self.is_debug:
            logger.info(f"save_df_as_pkl: {json_path=} {pkl_path=}")

        if self.is_manifold:
            buffer = read_data_from_manifold(json_path)
            buffer_txt = buffer.decode("utf-8")
            df = {}
            for i, line in enumerate(tqdm(buffer_txt.splitlines())):
                df[i] = eval(line)
            df = pd.DataFrame.from_dict(df, orient="index")
            buffer = io.BytesIO()
            dump(df, buffer)
            buffer.seek(0)
            save_data_to_manifold(pkl_path, buffer)

        else:
            with open(json_path, "r") as fin:
                df = {}
                for i, line in enumerate(tqdm(fin)):
                    df[i] = eval(line)
                df = pd.DataFrame.from_dict(df, orient="index")
            df.to_pickle(pkl_path)

    def create_dir(self, dst: str):
        if self.is_debug:
            logger.info(f"create_dir {dst=}")
        if self.is_manifold:
            create_manifold_directory(dst)
        else:
            os.makedirs(dst, exist_ok=True)

    def list_files_in_dir(self, path: str) -> List[str]:
        if self.is_manifold:
            return list_all_files(path)
        else:
            return os.listdir(path)
