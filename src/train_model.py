import logging
import os
import os.path as osp
import time
from glob import glob

import hydra
import pytorch_lightning as pl
import torch
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    BackboneFinetuning,
    GPUStatsMonitor,
    LearningRateMonitor,
)
from src.dataset_utils import get_datasets, TransformBatch, create_on_box_dataloader
from src.lit_utils import LitModel
from src.manifold_utils import save_data_to_manifold
from stl.lightning.callbacks.model_checkpoint import ModelCheckpoint
from stl.lightning.loggers.manifold_tensorboard_logger import ManifoldTensorBoardLogger

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
    logger.info(f"{manifold_out_path=}")
    monitor = "acc/val" if cfg.is_debug is False else "acc/train"
    checkpoint_callback = ModelCheckpoint(
        dirpath=manifold_out_path,
        has_user_data=False,
        monitor=monitor,
        save_top_k=1,
        mode="max",
    )
    logger.info(
        f"Tensorboard in https://internalfb.com/intern/tensorboard/?dir={manifold_out_path}"
    )

    # Load data
    t0 = time.time()
    train_batch_transform = TransformBatch(
        is_train=True,
        img_transform_crop=cfg.img_transform_crop,
        img_transform_resize=cfg.img_transform_resize,
    )
    test_batch_transform = TransformBatch(
        is_train=False,
        img_transform_crop=cfg.img_transform_crop,
        img_transform_resize=cfg.img_transform_resize,
    )

    train_dataset, test_dataset, dataset_meta = get_datasets(
        category=cfg.category,
        data_dir=cfg.data_dir,
        cf_vector_base_dir=cfg.cf_vector_base_dir,
        is_use_cf_bias=cfg.is_use_cf_bias,
        labeled_ratio=cfg.labeled_ratio,
        batch_size=cfg.batch_size,
        train_set_repeat=cfg.train_set_repeat,
        num_workers=cfg.num_workers,
        train_set_ratio=cfg.train_set_ratio,
        local_dir_path=cfg.local_dir_path,
        train_batch_transform=train_batch_transform,
        test_batch_transform=test_batch_transform,
    )
    logger.info(f"Loadded data in {time.time() -t0 :.2f} sec")
    logger.info(
        "Sizes [trainset testset num_classes]=[{} {} {}]".format(
            dataset_meta["train_set_size"],
            dataset_meta["test_set_size"],
            len(dataset_meta["classes"]),
        )
    )

    # Create dataloder
    t0 = time.time()
    trainloader = create_on_box_dataloader(
        dataset=train_dataset,
        num_workers=cfg.num_workers,
        dpp_server_num_worker_threads=cfg.dpp_server_num_worker_threads,
        phase="train",
    )
    testloader = create_on_box_dataloader(
        dataset=test_dataset,
        num_workers=cfg.num_workers,
        dpp_server_num_worker_threads=cfg.dpp_server_num_worker_threads,
        phase="test",
    )
    logger.info(f"create_on_box_dataloader in {time.time() -t0 :.2f} sec")

    # Load model
    lit_h = LitModel(len(dataset_meta["classes"]), dataset_meta["cf_vector_dim"], cfg)
    trainer = pl.Trainer(
        min_epochs=cfg["epochs"],
        max_epochs=cfg["epochs"],
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            GPUStatsMonitor(),
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
    train_model()
