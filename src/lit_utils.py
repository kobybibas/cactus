import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from multitask_utils import GradCosine
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from architecture_utils import get_backbone, get_cf_predictor, get_classifier

logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int,
        cf_vector_dim: int,
        cfg,
        pos_weight=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_target_classes = num_target_classes
        self.save_hyperparameters()
        self.map_best = 0

        pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        # Define the archotecture
        self.backbone, out_feature_num = get_backbone(cfg["is_pretrained"])
        self.classifier = get_classifier(out_feature_num, num_target_classes)
        self.cf_layers = get_cf_predictor(out_feature_num, cf_vector_dim)

        # For MTL
        self.weighting_method = GradCosine(main_task=0.0)
        if hasattr(self.cfg, "use_grad_cosine") and self.cfg.use_grad_cosine is True:
            self.automatic_optimization = False

    def criterion_cf(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cf_topk_loss_ratio: float,
        cf_confidence: torch.tensor,
    ):
        # Dim is as the len of the batch. the mean is for the cf vector dimentsion (64).
        loss_reduction_none = nn.functional.mse_loss(
            pred, target, reduction="none"
        ).mean(dim=-1) + nn.functional.l1_loss(pred, target, reduction="none").mean(
            dim=-1
        )
        loss_reduction_none = torch.exp(loss_reduction_none) - 1.0
        loss_reduction_none = (
            cf_confidence * loss_reduction_none / (cf_confidence.sum())
        )

        # Take item_num with lowest loss
        if cf_topk_loss_ratio < 1.0:
            num_cf_items = int(cf_topk_loss_ratio * len(loss_reduction_none))
            loss = torch.topk(
                loss_reduction_none, k=num_cf_items, largest=False, sorted=False
            )[0].sum()
        else:
            loss = loss_reduction_none.sum()

        return loss

    def criterion_cf_triplet(self, cf_hat: torch.Tensor, cf_hat_pos: torch.Tensor):
        cf_hat_neg = torch.roll(cf_hat_pos, shifts=1, dims=0)
        loss = nn.functional.triplet_margin_loss(
            cf_hat, cf_hat_pos, cf_hat_neg, margin=0.2, p=2
        )
        return loss

    def forward(self, x):
        representations = self.backbone(x).flatten(1)
        y_hat = self.classifier(representations)
        cf_hat = self.cf_layers(representations)
        return y_hat, cf_hat

    def _loss_helper(self, batch, phase: str):
        assert phase in ["train", "val"]

        (
            imgs,
            imgs_pos,
            cf_vectors,
            labels,
            is_labeled,
            cf_confidence,
        ) = batch
        y_hat, cf_hat = self(imgs)

        # Compute calssification loss
        loss_calssification = self.criterion(y_hat.squeeze(), labels.float().squeeze())
        loss_calssification = loss_calssification[is_labeled].mean()

        # Compute CF loss
        cf_topk_loss_ratio = self.cfg["cf_topk_loss_ratio"] if phase == "train" else 1.0
        if self.cfg.cf_loss_type == "exp":
            loss_cf = self.criterion_cf(
                cf_hat,
                cf_vectors,
                cf_topk_loss_ratio,
                cf_confidence,
            )
        elif self.cfg.cf_loss_type == "triplet":
            _, cf_hat_pos = self(imgs_pos)
            loss_cf = self.criterion_cf_triplet(cf_hat.squeeze(), cf_hat_pos.squeeze())
        else:
            raise ValueError(f"{self.cfg.cf_loss_type=}")

        # Combine loss
        loss = (
            self.cfg["label_weight"] * loss_calssification
            + self.cfg["cf_weight"] * loss_cf
        )
        if phase == "train":
            self.manual_backward(loss_calssification, loss_cf)

        res_dict = {
            f"loss/{phase}": loss.detach(),
            f"loss_classification/{phase}": loss_calssification.detach(),
            f"loss_cf/{phase}": loss_cf.detach(),
        }
        self.log_dict(
            res_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        preds = torch.sigmoid(y_hat)
        res_dict["labels"] = labels.cpu().detach().numpy()
        res_dict["preds"] = preds.cpu().detach().numpy()
        res_dict["loss"] = loss
        return res_dict

    def epoch_end_helper(self, outputs, phase: str):
        assert phase in ["train", "val"]
        loss = torch.mean(torch.stack([out["loss"] for out in outputs])).item()

        preds = np.vstack([out["preds"] for out in outputs])
        labels = np.vstack([out["labels"] for out in outputs])

        # Metrics
        auroc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        f1 = f1_score(labels, preds > 0.5, average="macro")

        # recall@k
        recall_dict, hit_dict = {}, {}
        for k in self.cfg["recall_at_k"]:
            recall_rate = self.compute_recall_at_k(preds, labels, k=k)
            recall_dict[f"recall@{k}/{phase}"] = recall_rate

        self.log_dict(
            {
                f"auroc/{phase}": auroc,
                f"ap/{phase}": ap,
                f"f1/{phase}": f1,
            },
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )
        self.log_dict(
            {**recall_dict, **hit_dict},
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

        loss, auroc, ap, f1 = np.round([loss, auroc, ap, f1], 3)
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] {phase} epoch end. {[loss, auroc, ap, f1]=}"
        )

        if phase == "val" and ap > self.map_best:
            self.map_best = ap

    def training_step(self, batch, batch_idx):
        return self._loss_helper(batch, phase="train")

    def training_epoch_end(self, outputs):
        self.epoch_end_helper(outputs, "train")

    def validation_step(self, batch, batch_idx, stage=None):
        return self._loss_helper(batch, phase="val")

    def validation_epoch_end(self, outputs):
        self.epoch_end_helper(outputs, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg["milestones"]
        )
        return [optimizer], [lr_scheduler]

    def compute_recall_at_k(self, preds: np.ndarray, labels: np.ndarray, k: int = 5):
        recall_sum, item_num = 0, 0
        for pred, label in zip(torch.tensor(preds), torch.tensor(labels)):
            _, pred_idx = torch.topk(pred, k=k)  # The predicted labels
            label_idx = torch.where(label == 1)[0]  # The ground truth labels

            # In case there are no labels
            if len(label_idx) == 0:
                continue

            # Recal per item
            recall_i = sum(el in pred_idx for el in label_idx) / len(label_idx)

            recall_sum += recall_i
            item_num += 1

        # Average recall
        recall_rate = recall_sum / item_num
        return recall_rate

    def manual_backward(self, loss_classification, loss_cf):
        self._verify_is_manual_optimization("manual_backward")
        # self.trainer.accelerator.backward(loss, None, None, *args, **kwargs)

        opt = self.optimizers()
        opt.zero_grad()

        # Weight losses and backward
        shared_parameters = [p for _, p in self.backbone.named_parameters()]
        self.weighting_method.backward(
            [loss_classification, loss_cf],
            shared_parameters=shared_parameters,
            retain_graph=False,
        )

        # Update parameters
        opt.step()
        # opt.zero_grad()

    # def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
    #     pass


class LitModelCFBased(LitModel):
    def __init__(
        self,
        num_target_classes: int,
        cf_vector_dim: int,
        cfg,
        pos_weight=None,
    ):
        # pl.LightningModule().__init__()
        cfg["is_pretrained"] = False
        super().__init__(num_target_classes, cf_vector_dim, cfg, pos_weight)

        # Define the backbone
        layer_dims = torch.linspace(cf_vector_dim, num_target_classes, 3).int()
        self.model = nn.Sequential(
            nn.BatchNorm1d(layer_dims[0]),
            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_dims[1], layer_dims[2]),
        )
        logger.info(self)

    def forward(self, x):
        return self.model(x)

    def _loss_helper(self, batch, phase: str):
        assert phase in ["train", "val"]

        cf_vectors, labels = batch
        y_hat = self(cf_vectors)

        # Compute calssification loss
        loss = self.criterion(y_hat.squeeze(), labels.float().squeeze()).mean()

        res_dict = {
            f"loss/{phase}": loss.detach(),
        }
        self.log_dict(
            res_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        preds = torch.sigmoid(y_hat)
        res_dict["labels"] = labels.cpu().detach().numpy()
        res_dict["preds"] = preds.cpu().detach().numpy()
        res_dict["loss"] = loss

        # Most pop prediction as baseline
        if False:
            popolarity = np.array(
                self.train_dataloader.dataloader.dataset.df.label_vec.tolist()
            ).sum(axis=0)
            freq = popolarity / len(self.train_dataloader.dataloader.dataset.df)
            res_dict["preds"] = freq[np.newaxis, :].repeat(
                res_dict["preds"].shape[0], 0
            )
        return res_dict
