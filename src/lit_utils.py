import logging
import types

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models

logger = logging.getLogger(__name__)


def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    y_hat = self.fc(x)

    # CF layer
    cf_hat = self.cf_layers(x)

    return y_hat, cf_hat


def get_resetnet18(num_classes: int, is_pretrained: bool = False):
    resnet18 = models.resnet18(pretrained=is_pretrained)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)

    # Adding prediction of CF vector
    cf_layers = nn.Sequential(
        nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
    )
    resnet18.cf_layers = cf_layers
    resnet18._forward_impl = types.MethodType(_forward_impl, resnet18)
    return resnet18


class LitModel(pl.LightningModule):
    def __init__(
        self, num_target_classes: int, cf_vector_dim: int, cfg, label_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.cfg = cfg
        self.train_acc_metric, self.val_acc_metric = (
            torchmetrics.Accuracy(),
            torchmetrics.Accuracy(),
        )  # Explicit define in  self so lightning handle device
        self.acc_metric = {
            "train": self.train_acc_metric,
            "val": self.val_acc_metric,
        }
        self.label_weights = label_weights

        # Define the backbone
        backbone = models.resnet18(pretrained=True)
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
        logger.info(self)

    def criterion_cf(self, pred, target, cf_topk_loss_ratio: float):
        # Dim is as the len of the batch. the mean is for the cf vector dimentsion (32/33).
        loss_reduction_none = nn.functional.mse_loss(
            pred, target, reduction="none"
        ).mean(dim=-1)

        # Take items with lowest loss
        num_cf_items = int(cf_topk_loss_ratio * len(loss_reduction_none))
        loss = torch.topk(
            loss_reduction_none, k=num_cf_items, largest=False, sorted=False
        )[0].mean()
        return loss, num_cf_items

    def forward(self, x):
        representations = self.backbone(x).flatten(1)
        y_hat = self.classifier(representations)
        cf_hat_hat = self.cf_layers(representations)
        return y_hat, cf_hat_hat

    def _loss_helper(self, batch, phase: str):
        assert phase in ["train", "val"]

        imgs, labels, is_labeled, cf_vectors = batch
        y_hat, cf_hat = self(imgs)

        # Compute calssification loss
        loss_calssification = self.criterion(y_hat.squeeze(), labels.squeeze())[
            is_labeled
        ].mean()

        # Compute cf loss Take item with lowest loss
        cf_topk_loss_ratio = self.cfg.cf_topk_loss_ratio if phase == "train" else 1.0
        loss_cf, num_cf_items = self.criterion_cf(
            cf_hat.squeeze(), cf_vectors.squeeze(), cf_topk_loss_ratio
        )
        cf_hat_var = cf_hat.var(dim=-1).mean()

        # Combine loss
        loss = loss_calssification + self.cfg.cf_weight * loss_cf
        if False:  # TODO: loss weight based on labels
            if self.label_weights is not None:
                weight_batch = self.label_weights[labels]
                weight_batch /= weight_batch.sum()
                loss = (loss * weight_batch).sum()
            else:
                loss = loss.mean()

        preds = torch.argmax(y_hat, dim=1)
        acc = self.acc_metric[phase](preds, labels).item()

        res_dict = {
            "loss": loss,
            f"loss/{phase}": loss,
            f"acc/{phase}": acc,
            f"num_imgs/{phase}": len(imgs),
            f"loss_classification/{phase}": loss_calssification,
            f"loss_cf/{phase}": loss_cf,
            f"num_cf_items/{phase}": num_cf_items,
            f"is_labeled_num/{phase}": is_labeled.sum(),
            f"cf_hat_var/{phase}": cf_hat_var,
        }
        self.log_dict(
            res_dict,
            prog_bar=True,
            logger=True,
            on_step=phase == "train",
            on_epoch=True,
        )
        return res_dict

    def epoch_end_helper(self, outputs, phase: str):
        assert phase in ["train", "val"]
        loss = torch.mean(torch.stack([out["loss"] for out in outputs])).item()
        acc = self.acc_metric[phase].compute().item()
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] {phase} epoch end. {[loss, acc]=}"
        )

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
