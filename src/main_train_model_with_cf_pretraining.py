import logging
import os
import time
import os.path as osp
import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from hydra.core.hydra_config import HydraConfig
from dataset_utils import get_datasets
from lit_utils import LitModel

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="train_model_with_cf_pretraining",
)
def train_model_with_cf_pretraining(cfg: DictConfig):
    t_start = time.time()
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(get_original_cwd())
    logger.info(f"{out_dir=}")
    pl.utilities.seed.seed_everything(cfg.seed)
    logger.info(f"{torch.cuda.is_available()=}")

    # Configure logging
    tb_logger = pl_loggers.TensorBoardLogger(out_dir)
    tb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Load data
    t0 = time.time()
    train_dataset, test_dataset, dataset_meta, pos_weight = get_datasets(
        cfg.train_df_path,
        cfg.test_df_path,
        cfg.cf_vector_df_path,
        cfg.labeled_ratio,
    )
    logger.info(f"Loadded data in {time.time() -t0 :.2f} sec")
    logger.info(
        "Sizes [trainset testset num_classes]=[{} {} {}]".format(
            dataset_meta["train_set_size"],
            dataset_meta["test_set_size"],
            dataset_meta["num_classes"],
        )
    )

    # Create dataloder
    t0 = time.time()
    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Load model
    lit_h = LitModel(
        dataset_meta["num_classes"],
        dataset_meta["cf_vector_dim"],
        cfg["cf_training"],
        pos_weight,
    )
    trainer = pl.Trainer(
        min_epochs=cfg["cf_training"]["epochs"],
        max_epochs=cfg["cf_training"]["epochs"],
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
        ],
        fast_dev_run=cfg.is_debug,
        num_sanity_val_steps=0,
        gpus=[cfg.gpu] if torch.cuda.is_available() else None,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish cf training in {time.time() -t_start :.2f} sec")
    logger.info(f"{out_dir=}")
    trainer.save_checkpoint(osp.join(out_dir, "model_pretrained_cf.ckpt"))

    # Train labeled
    lit_h.cfg = cfg["label_training"]
    trainer = pl.Trainer(
        min_epochs=cfg["label_training"]["epochs"],
        max_epochs=cfg["label_training"]["epochs"],
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
        ],
        fast_dev_run=cfg.is_debug,
        num_sanity_val_steps=0,
        gpus=[cfg.gpu] if torch.cuda.is_available() else None,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish label training in {time.time() -t_start :.2f} sec")
    logger.info(f"{out_dir=}")
    trainer.save_checkpoint(osp.join(out_dir, "model.ckpt"))


if __name__ == "__main__":
    train_model_with_cf_pretraining()
