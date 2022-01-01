import logging
import os
import os.path as osp
import time

import re
import urllib.request
import hydra
import pandas as pd
from tqdm import tqdm

from data_utils import DataHelper

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="download_flicker_set",
)
def download_data(cfg):
    t0 = time.time()

    logger.info(cfg)
    logger.info(f"{os.getcwd()=}")

    data_helper = DataHelper(is_debug=cfg.is_debug, is_override=cfg.is_override)
    data_helper.create_dir(cfg.data_dir)

    # Dowload category data
    t1 = time.time()
    dst_download_path = osp.join(cfg.data_dir, f"flicker.tsv.zip")
    data_helper.download_url(cfg.base_url, dst_download_path)
    logger.info(f"Download tsv in {time.time()-t1:.2f} sec")

    t1 = time.time()
    dst_path = osp.join(cfg.data_dir, f"datastore")
    dst_unzip_path = osp.join(cfg.data_dir, "datastore", "added_tags_tilMarch2016.tsv")
    os.system(f"unzip -o {dst_download_path} -d {cfg.data_dir}")
    dst_path = osp.join(cfg.data_dir, f"flicker.tsv")
    os.replace(dst_unzip_path, dst_path)
    logger.info(f"Unzip tsv.zip in {time.time()-t1:.2f} sec")

    df = pd.read_csv(dst_path, delimiter="\t")
    df["imUrl"] = cfg.img_base_url + df["flickrid"].astype(str)
    logger.info(f"Load df in {time.time()-t1:.2f} sec")

    df = df.dropna()

    # Create recommendation
    out_path =osp.join(cfg.data_dir, f'rating_flicker_user_based.txt')
    df["overall"] = 5.0  # Align with Amazon datasets
    df[['author','flickrid','overall']].to_csv(out_path, sep="\t", index=False, header=False)

    # pd.value_counts(df['tag'])

    # Create output directory for images
    categrory_dir = osp.join(cfg.data_dir, "flicker_images")
    data_helper.create_dir(categrory_dir)
    logger.info(f"{categrory_dir=}")

    # Download
    img_urls = df.drop_duplicates(subset="imUrl", keep="first")["imUrl"]
    logger.info(f"{len(img_urls)=}")

    t1 = time.time()
    dst_exsits = data_helper.list_files_in_dir(categrory_dir)
    dst_imgs = [
        osp.join(categrory_dir, img_url[img_url.find("=") + 1 :] + ".jpg")
        for img_url in tqdm(img_urls)
    ]
    logger.info(f"{len(dst_exsits)=} in {time.time()-t1:.2f} sec")

    # Filter lists for exists
    dst_imgs_filt, img_urls_filt = [], []
    for dst, url in zip(dst_imgs, img_urls):
        if osp.basename(dst) not in dst_exsits:
            img_urls_filt.append(url)
            dst_imgs_filt.append(dst)
    assert len(img_urls_filt) == len(dst_imgs_filt)
    logger.info(f"Post filtering. {len(img_urls_filt)=}")

    pattern = re.compile(cfg.img_html_match)
    for img_url, dst_img in tqdm(
        zip(img_urls_filt, dst_imgs_filt), total=len(img_urls_filt)
    ):
        # get url to download
        try:
            website = urllib.request.urlopen(img_url)
            html = str(website.read())
            img_url_to_download = pattern.search(html).group()
            data_helper.download_url(img_url_to_download, dst_img)
        except:
            logger.info(f'Failed {img_url=}')
    logger.info(f"Finish in {time.time() -t0:.2f} sec")


if __name__ == "__main__":
    download_data()
