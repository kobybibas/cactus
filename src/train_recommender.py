import logging
import os
import os.path as osp
import time

import cornac
import hydra
import pandas as pd
import torch
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC
from cornac.models import VAECF
from omegaconf import DictConfig

from recommender_utils import AmazonClothing, HitRate

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="train_recommender",
)
def train_recommender(cfg: DictConfig):
    out_dir = os.getcwd()
    logger.info(cfg)
    logger.info(os.getcwd())

    # Initalize dataset
    dataset_h = AmazonClothing(cfg.data_dir,cfg.category)
    dataset = dataset_h.load_feedback()
    rs = RatioSplit(
        data=dataset,
        test_size=cfg.test_size,
        rating_threshold=1.0,
        seed=cfg.seed,
        exclude_unknowns=True,
        verbose=True,
    )

    # Initalize model
    vaecf = VAECF(
        k=cfg.bottleneck_size,
        autoencoder_structure=list(cfg.emb_size),
        act_fn="tanh",
        likelihood="mult",
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        beta=cfg.beta,
        seed=cfg.seed,
        use_gpu=True,
        verbose=True,
    )

    # Run training
    t0 = time.time()
    metrics = [AUC()] + [HitRate(k=top_k) for top_k in cfg.top_k_list]
    cornac.Experiment(
        eval_method=rs, models=[vaecf], metrics=metrics, user_based=False
    ).run()

    logger.info(f"Finish training in {time.time() -t0:.2f} sec")
    logger.info(vaecf.vae)

    # Save item embeddings
    out_path = osp.join(out_dir, "item_emb.pt")
    logger.info(out_path)
    torch.save(vaecf.vae.decoder.fc1.weight.detach().cpu(), out_path)

    embs = vaecf.vae.decoder.fc1.weight.detach().cpu()
    df = pd.DataFrame({"asin": list(rs.train_set.item_ids), "embs": embs.tolist()})

    out_path = osp.join(out_dir, "cf_df.pkl")
    logger.info(out_path)
    df.to_pickle(out_path)


if __name__ == "__main__":
    train_recommender()
