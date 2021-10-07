import logging
import os
import os.path as osp
import time

import hydra
from hydra.utils import get_original_cwd
import pytorch_lightning as pl
import torch
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    BackboneFinetuning,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning import loggers as pl_loggers
from dataset_utils import get_datasets
from lit_utils import LitModel
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)


class BackboneFinetuningWithLogs(BackboneFinetuning):
    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer,
        opt_idx: int,
    ):
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                logger.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.round)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = (
                self.lambda_func(epoch + 1) * self.previous_backbone_lr
            )
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                logger.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.round)}"
                )


@hydra.main(
    config_path="../configs",
    config_name="train_model",
)
def train_model(cfg: DictConfig):
    t_start = time.time()
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(get_original_cwd())
    logger.info(f"{out_dir=}")
    pl.utilities.seed.seed_everything(cfg.seed)
    logger.info(f"{torch.cuda.is_available()=}")

    # Configure logging
    unique_out_dir = osp.basename(os.getcwd())
    tb_logger = pl_loggers.TensorBoardLogger(out_dir)
    tb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Configure checkpoint saver
    monitor = "auroc/val" if cfg.is_debug is False else "auroc/train"
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        monitor=monitor,
        save_top_k=1,
        mode="max",
    )
    logger.info(f"Tensorboard in https://internalfb.com/intern/tensorboard/?dir=")

    # Load data
    t0 = time.time()
    train_dataset, test_dataset, dataset_meta = get_datasets(
        cfg.label_df_path,
        cfg.cf_vector_df_path,
        cfg.labeled_ratio,
        cfg.train_set_ratio,
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
    lit_h = LitModel(dataset_meta["num_classes"], dataset_meta["cf_vector_dim"], cfg)
    trainer = pl.Trainer(
        min_epochs=cfg["epochs"],
        max_epochs=cfg["epochs"],
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            BackboneFinetuningWithLogs(
                unfreeze_backbone_at_epoch=cfg.unfreeze_backbone_at_epoch,
                # lambda_func=lambda epoch: cfg.backbone_lr_multiplicative,
                verbose=True,
                # backbone_initial_lr=cfg.backbone_initial_lr,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        fast_dev_run=cfg.is_debug,
        num_sanity_val_steps=0,
        gpus=[cfg.gpu] if torch.cuda.is_available() else None,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish training in {time.time() -t_start :.2f} sec")
    logger.info(f"{os.getcwd()=}")


if __name__ == "__main__":
    train_model()
