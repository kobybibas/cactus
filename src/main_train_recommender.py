import logging
import os
import os.path as osp
import time

import cornac
import hydra
import pandas as pd
import torch
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC, MAP
from omegaconf import DictConfig
from recommender_utils import AmazonClothing
from vae_utils import VAECFWithBias

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
    dataset_h = AmazonClothing(cfg.data_dir, cfg.category, cfg.user_based)
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
    most_pop = cornac.models.MostPop()
    vaecf = VAECFWithBias(
        k=cfg.bottleneck_size,
        autoencoder_structure=list(cfg.emb_size),
        act_fn="tanh",
        likelihood="mult",
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        lr_steps=cfg.lr_steps,
        beta=cfg.beta,
        seed=cfg.seed,
        use_gpu=True,
        verbose=True,
        out_dir=out_dir
    )

    # Run training
    t0 = time.time()
    metrics = [AUC(), MAP()]
    cornac.Experiment(
        eval_method=rs, models=[most_pop, vaecf], metrics=metrics, user_based=False
    ).run()

    logger.info(f"Finish training in {time.time() -t0:.2f} sec")
    logger.info(vaecf.vae)

    # Save vae model
    out_path = osp.join(out_dir, "vae.pt")
    torch.save(vaecf.vae.state_dict(), out_path)

    # Create CF data frame
    num_intercations = rs.train_set.csc_matrix.sum(axis=0).tolist()[0]
    embs = vaecf.vae.decoder.fc1.weight.detach().cpu()
    bias = vaecf.vae.item_bias.weight.detach().cpu().squeeze()
    df = pd.DataFrame(
        {
            "asin": list(rs.train_set.item_ids),
            "embs": embs.tolist(),
            "bias": bias.tolist(),
            "num_intercations": num_intercations,
        }
    )

    # Save to: out path
    out_path = osp.join(out_dir, "cf_df.pkl")
    logger.info(out_path)
    df.to_pickle(out_path)

    # Save to: dataset output top dir
    out_path = osp.join(out_dir, "..", "cf_df.pkl")
    logger.info(out_path)
    df.to_pickle(out_path)


if __name__ == "__main__":
    train_recommender()