import logging
import os
import os.path as osp
import time
from glob import glob

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import (
    ModelCheckpoint as OSSModelCheckpoint,
)
from src.bpr_utils import BprModel, BprLitModel, get_bpr_datasets
from src.manifold_utils import save_data_to_manifold
from stl.lightning.callbacks.model_checkpoint import ModelCheckpoint
from stl.lightning.loggers.manifold_tensorboard_logger import ManifoldTensorBoardLogger
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="train_bpr",
)
def train_bpr(cfg: DictConfig):
    t_start = time.time()
    logger.info(cfg)
    logger.info(f"{os.getcwd()=}")
    pl.utilities.seed.seed_everything(cfg.seed)
    logger.info(f"{torch.cuda.is_available()=}")

    # Configure logging
    manifold_path_split = cfg.manifold_out_dir.split("/")
    unique_out_dir = osp.basename(os.getcwd())
    tb_logger = ManifoldTensorBoardLogger(
        manifold_bucket=manifold_path_split[0],
        manifold_path="/".join(manifold_path_split[1:]),
        name=unique_out_dir,
    )
    tb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Configure checkpoint saver
    manifold_out_path = osp.join(f"manifold://{cfg.manifold_out_dir}", unique_out_dir)
    monitor = "acc/val" if cfg.is_debug is False else "acc/train"
    checkpoint_callback = ModelCheckpoint(
        dirpath=manifold_out_path, has_user_data=False, monitor=monitor, save_top_k=1
    )
    checkpoint_callback_local = OSSModelCheckpoint(
        dirpath=os.getcwd(), monitor=monitor, save_top_k=1
    )

    logger.info(
        f"Tensorboard in https://internalfb.com/intern/tensorboard/?dir={manifold_out_path}"
    )

    # Load data
    t0 = time.time()
    train_dataset, test_dataset = get_bpr_datasets(
        cfg.data_dir,
        cfg.category,
        cfg.local_dir_path,
        device="cpu" if not torch.cuda.is_available() else f"cuda:{cfg.gpu}",
    )
    trainloader = DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        test_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    logger.info(
        f"{len(train_dataset)=} {len(test_dataset)=}. In {time.time() -t0 :.2f} sec"
    )

    # Load model
    model = BprModel(
        train_dataset.num_users,
        train_dataset.num_items,
        cfg.emb_size,
        cfg.embedding_max_norm,
    )
    lit_h = BprLitModel(model, cfg)
    trainer = pl.Trainer(
        min_epochs=cfg["epochs"],
        max_epochs=cfg["epochs"],
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            checkpoint_callback,
            checkpoint_callback_local,
            GPUStatsMonitor(),
        ],
        fast_dev_run=cfg.is_debug,
        num_sanity_val_steps=0,
        gpus=[cfg.gpu] if torch.cuda.is_available() else None,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish training in {time.time() -t_start :.2f} sec")

    # Load best checkpoint
    logger.info(f"Loading best checkpoint {checkpoint_callback_local.best_model_path=}")
    lit_h = BprLitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback_local.best_model_path
    )

    # Save embeddings
    (
        embed_users,
        embed_items,
        embed_items_bias,
    ) = lit_h.model.get_embedding_weights()

    save_data_to_manifold(
        osp.join(f"{cfg.manifold_out_dir}", unique_out_dir, "embed_users.pt"),
        embed_users,
        is_pkl=True,
    )
    save_data_to_manifold(
        osp.join(f"{cfg.manifold_out_dir}", unique_out_dir, "embed_items.pt"),
        embed_items,
        is_pkl=True,
    )
    save_data_to_manifold(
        osp.join(f"{cfg.manifold_out_dir}", unique_out_dir, "embed_items_bias.pt"),
        embed_items_bias,
        is_pkl=True,
    )

    logger.info(f"Saved embeddings in {time.time() -t0 :.2f} sec")

    lit_h.calc_topk(
        dataset=test_dataset, top_k_list=cfg.top_k_list, tb_logger=tb_logger
    )

    # Move logs to manifold
    log_file = glob(osp.join(os.getcwd(), "*.log"))[0]
    with open(log_file, "rb") as f:
        buf = f.read()
    save_data_to_manifold(
        osp.join(f"{cfg.manifold_out_dir}", unique_out_dir, "output.log"),
        buf,
        is_pkl=False,
    )
    logger.info(f"{os.getcwd()=}")


if __name__ == "__main__":
    train_bpr()
