import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

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

        pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        # Define the backbone
        backbone = models.resnet18(pretrained=cfg["is_pretrained"])
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)

        # Define the classification layer
        num_filters = backbone.fc.in_features
        self.classifier = nn.Linear(num_filters, num_target_classes)

        # Define the cf vector predictor
        self.cf_layers = nn.Sequential(
            nn.BatchNorm1d(num_filters),
            nn.Linear(num_filters, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, cf_vector_dim),
        )
        # self.cf_layers = nn.Sequential(
        #     nn.BatchNorm1d(num_filters),
        #     nn.Linear(num_filters, cf_vector_dim),
        # )
        logger.info(self)

    def criterion_cf(self, pred, target, cf_topk_loss_ratio: float):
        # Dim is as the len of the batch. the mean is for the cf vector dimentsion (64).
        loss_reduction_none = nn.functional.mse_loss(
            pred, target, reduction="none"
        ).mean(dim=-1) + nn.functional.l1_loss(pred, target, reduction="none").mean(
            dim=-1
        )
        loss_reduction_none = torch.exp(loss_reduction_none) - 1.0 # TODO: experimneting with exp

        # Take items with lowest loss
        if cf_topk_loss_ratio < 1.0:
            num_cf_items = int(cf_topk_loss_ratio * len(loss_reduction_none))
            loss = torch.topk(
                loss_reduction_none, k=num_cf_items, largest=False, sorted=False
            )[0].mean()
        else:
            num_cf_items = len(loss_reduction_none)
            loss = loss_reduction_none.mean()
        return loss, num_cf_items

    # def criterion_cf(self, pred, target, cf_topk_loss_ratio: float):
    #     # Dim is as the len of the batch. the mean is for the cf vector dimentsion (64).
    #     # loss = nn.CosineSimilarity(dim=1, eps=1e-6)(
    #     #     pred, target
    #     # ).abs().mean(dim=-1)
    #     loss = (pred*target).sum(axis=1).abs().mean()
    #     num_cf_items = len(pred)
    #     return loss, num_cf_items

    def forward(self, x):
        representations = self.backbone(x).flatten(1)
        y_hat = self.classifier(representations)
        cf_hat_hat = self.cf_layers(representations)
        return y_hat, cf_hat_hat

    def _loss_helper(self, batch, phase: str):
        assert phase in ["train", "val"]

        (
            imgs,
            cf_vectors,
            labels,
            is_labeled,
        ) = batch
        y_hat, cf_hat = self(imgs)

        # Compute calssification loss
        loss_calssification = self.criterion(y_hat.squeeze(), labels.float().squeeze())
        loss_calssification = loss_calssification[is_labeled].mean()

        # Compute cf loss Take item with lowest loss
        cf_topk_loss_ratio = self.cfg["cf_topk_loss_ratio"] if phase == "train" else 1.0
        loss_cf, num_cf_items = self.criterion_cf(
            cf_hat.squeeze(), cf_vectors.squeeze(), cf_topk_loss_ratio
        )
        cf_hat_var = cf_hat.var(dim=-1).mean()

        # Combine loss
        loss = loss_calssification + self.cfg["cf_weight"] * loss_cf

        res_dict = {
            f"loss/{phase}": loss.detach(),
            f"loss_classification/{phase}": loss_calssification.detach(),
            f"loss_cf/{phase}": loss_cf.detach(),
            f"num_cf_items/{phase}": num_cf_items,
            f"is_labeled_num/{phase}": is_labeled.sum(),
            f"cf_hat_var/{phase}": cf_hat_var.detach(),
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
        acc = accuracy_score(labels, preds > 0.5)
        f1 = f1_score(labels, preds > 0.5, average="macro")

        # recall@k
        recall_dict,hit_dict = {}, {}
        for k in self.cfg.recall_at_k:
            recall_rate, hit_rate = self.compute_recall_and_hit_rate_at_k(
                preds, labels, k=k
            )
            recall_dict[f"recall@{k}/{phase}"] = recall_rate
            hit_dict[f"hit_rate@{k}/{phase}"] = hit_rate

        self.log_dict(
            {
                f"auroc/{phase}": auroc,
                f"ap/{phase}": ap,
                f"acc/{phase}": acc,
                f"f1/{phase}": f1,
            },
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )
        self.log_dict(
            {**recall_dict, **hit_dict} ,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

        loss, acc, auroc, ap, f1 = np.round([loss, acc, auroc, ap, f1], 3)
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] {phase} epoch end. {[loss,acc, auroc, ap, f1]=}"
        )
        logger.info([f'{key}={np.round(value,3)}' for key, value in recall_dict.items()])

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

    def compute_recall_and_hit_rate_at_k(self,preds, labels, k: int = 5):
        recall_sum, hit_sum = 0, 0
        items = 0
        for pred, label in zip(torch.tensor(preds), torch.tensor(labels)):
            _, pred_idx = torch.topk(pred, k=k)
            label_idx = torch.where(label == 1)[0]

            if len(label_idx) == 0:
                continue

            recall_i = sum(el in pred_idx for el in label_idx) / len(label_idx)
            recall_sum += recall_i

            hit_i = sum(el in label_idx for el in pred_idx)
            hit_sum += hit_i

            items += 1

        recall_rate = recall_sum / items
        hit_rate = hit_sum / items
        return recall_rate, hit_rate
