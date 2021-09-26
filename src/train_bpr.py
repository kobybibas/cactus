import logging
import os
import time

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from bpr_utils import BprModel, BprLitModel, get_bpr_datasets
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

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
    tb_logger = TensorBoardLogger(os.getcwd(), name="bpr")
    tb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Configure checkpoint saver
    monitor = "acc/val" if cfg.is_debug is False else "acc/train"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), monitor=monitor, save_top_k=1, mode="max"
    )

    # Load data
    t0 = time.time()
    train_dataset, test_dataset = get_bpr_datasets(
        cfg.data_dir,
        cfg.category,
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
            GPUStatsMonitor(),
        ],
        fast_dev_run=cfg.is_debug,
        num_sanity_val_steps=0,
        gpus=[cfg.gpu] if torch.cuda.is_available() else None,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish training in {time.time() -t_start :.2f} sec")

    # Load best checkpoint
    logger.info(f"Loading best checkpoint {checkpoint_callback.best_model_path=}")
    lit_h = BprLitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path
    )

    # Save embeddings
    (
        embed_users,
        embed_items,
        embed_items_bias,
    ) = lit_h.model.get_embedding_weights()

    torch.save(embed_users, "embed_users.pt")
    torch.save(embed_items, "embed_items.pt")
    torch.save(embed_items_bias, "embed_items_bias.pt")
    logger.info(f"Saved embeddings in {time.time() -t0 :.2f} sec")

    lit_h.calc_topk(
        dataset=test_dataset, top_k_list=cfg.top_k_list, tb_logger=tb_logger
    )
    logger.info(f"{os.getcwd()=}")


if __name__ == "__main__":
    train_bpr()
