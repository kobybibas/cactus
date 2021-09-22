import logging
import os
import os.path as osp
import pickle
import time
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from src.manifold_utils import read_data_from_manifold
from torchmetrics.functional import recall, precision
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_bpr_datasets(
    data_dir: str, category: str, local_dir_path: str = None, device: str = "cpu"
):
    if local_dir_path is not None:
        os.makedirs(local_dir_path, exist_ok=True)
        local_pkl_path = osp.join(local_dir_path, f"{category}.pkl")
        if not osp.exists(local_pkl_path):
            pkl_data = read_data_from_manifold(
                f"{data_dir}/{category}.pkl",
                is_from_pkl=True,
            )
            logger.info(f"Finish read_data_from_manifold. {list(pkl_data.keys())=}")
            with open(local_pkl_path, "wb") as f:
                pickle.dump(pkl_data, f, pickle.HIGHEST_PROTOCOL)

        with open(local_pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
    else:
        pkl_data = read_data_from_manifold(
            f"{data_dir}/{category}.pkl",
            is_from_pkl=True,
        )

    train_set, test_set = pkl_data["train_set"], pkl_data["test_set"]
    num_users, num_items = pkl_data["user_count"], pkl_data["item_count"]
    logger.info(f"{num_users=} {num_items=}")
    train_set = BPRData(train_set, num_users, num_items, device=device)
    test_set = BPRData(test_set, num_users, num_items, device=device)
    return train_set, test_set


class BPRData(data.Dataset):
    def __init__(
        self, set_data: np.ndarray, num_users: int, num_items: int, device: str = "cpu"
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dataset_data = torch.from_numpy(set_data).to(device)

    def __len__(self):
        return len(self.dataset_data)

    def __getitem__(self, idx):
        user_id, item_i = self.dataset_data[idx]
        return user_id, item_i


class BprModel(nn.Module):
    def __init__(self, user_num: int, item_num: int, emb_size: int = 32, max_norm=1.0):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_user = nn.Embedding(user_num, emb_size, max_norm=max_norm)
        self.embed_item = nn.Embedding(item_num, emb_size, max_norm=max_norm)
        self.embed_item_bias = nn.Embedding(item_num, 1)

        nn.init.normal_(self.embed_user.weight, std=0.1)
        nn.init.normal_(self.embed_item.weight, std=0.1)
        nn.init.normal_(self.embed_item_bias.weight, std=0.01)

    def get_embedding_weights(self):
        user_weight = self.embed_user(torch.arange(self.user_num))
        item_weight = self.embed_item(torch.arange(self.item_num))
        item_bias_weight = self.embed_item_bias(torch.arange(self.item_num))
        return user_weight, item_weight, item_bias_weight

    def forward(self, user_idx, item_i_idx):
        user = self.embed_user(user_idx)
        item_i = self.embed_item(item_i_idx)
        item_bias_i = self.embed_item_bias(item_i_idx)

        batch_size = len(user)
        ones = torch.ones(batch_size, 1, device=user.device).float()
        user_w_ones = torch.hstack((user, ones))
        item_w_bias = torch.hstack((item_i, item_bias_i))

        prediction_i = (user_w_ones * item_w_bias).sum(dim=-1, keepdims=True)
        user_l2_norm = torch.linalg.norm(user, ord=2, dim=-1)
        item_i_l2_norm = torch.linalg.norm(item_w_bias, ord=2, dim=-1)
        return prediction_i, user_l2_norm, item_i_l2_norm


class BprLitModel(pl.LightningModule):
    def __init__(self, model: BprModel, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, user_idx, item_i_idx):
        return self.model(user_idx, item_i_idx)

    def calc_loss(self, pos: torch.Tensor, negs: torch.Tensor):
        # First dim is the positive, other are negeative
        predictions = torch.hstack((pos, torch.hstack(negs)))
        batch_size = len(predictions)
        targets = torch.zeros(batch_size, device=predictions.device).long()
        loss = self.cross_entropy(predictions, targets)
        return loss

    def calc_acc(self, pred_i, pred_j):
        return (pred_i > pred_j).float().mean()

    def _loss_helper(self, batch, phase: str):
        user_id, item_i = batch

        # Predict positive item
        pos, user_l2_norms, item_l2_norms = self(user_id, item_i)
        user_l2_norms, item_l2_norms = user_l2_norms.mean(), item_l2_norms.mean()

        negs = []
        shifts = torch.randint(
            low=1, high=len(item_i) - 1, size=(self.cfg.num_negatives,)
        ).int()
        for shift in shifts:
            item_j = torch.roll(item_i, shifts=shift.item(), dims=0)
            neg, user_l2_norm, item_l2_norm = self(user_id, item_j)
            negs.append(neg)
            item_l2_norms += item_l2_norm.mean()
            user_l2_norms += user_l2_norm.mean()
        l2_norm = user_l2_norms + item_l2_norms

        # Metrics
        loss_classification = self.calc_loss(pos, negs)
        loss = loss_classification + self.cfg.l2_reg * l2_norm
        acc = self.calc_acc(pos, negs[0])

        # Log
        self.log_dict(
            {
                f"loss/{phase}": loss,
                f"acc/{phase}": acc,
                f"l2_norm/{phase}": l2_norm,
                f"loss_classification/{phase}": loss_classification,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss, "acc": acc}

    def epoch_end_helper(self, outputs, phase: str):
        assert phase in ["train", "val"]
        loss = torch.mean(torch.tensor([out["loss"] for out in outputs])).item()
        acc = torch.mean(torch.tensor([out["acc"] for out in outputs])).item()
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] {phase} epoch end. [Loss Acc]=[{loss:.4f} {acc:.2f}]"
        )

    def training_step(self, batch, batch_idx):
        return self._loss_helper(batch, "train")

    def training_epoch_end(self, outputs):
        self.epoch_end_helper(outputs, "train")

    def validation_step(self, batch, batch_idx, stage=None):
        return self._loss_helper(batch, "val")

    def validation_epoch_end(self, outputs):
        self.epoch_end_helper(outputs, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.cfg["lr"],
            weight_decay=0.0,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg["milestones"]
        )
        return [optimizer], [lr_scheduler]

    def calc_topk(self, dataset: BPRData, top_k_list: List[int], tb_logger):
        # preds is a matrix: Each row represnets a user and each column is the score of the user to a product.
        embed_user, embed_item, embed_item_bias = self.model.get_embedding_weights()
        ones = torch.ones(len(embed_user), 1)
        embed_user = torch.hstack((embed_user, ones))
        embed_item = torch.hstack((embed_item, embed_item_bias))

        preds = embed_user @ embed_item.t()
        logger.info(
            f"{preds.shape=} {preds.device=} [embed_user embed_item.T]={embed_user.shape} {embed_item.t().shape}"
        )

        # Evaluate precision for various k
        hit_rate = {top_k: [] for top_k in top_k_list}
        topk_max = torch.max(torch.tensor(top_k_list))
        for user_id, item_id in tqdm(dataset):
            user_preds = preds[user_id]
            _, user_top_item_ids = torch.topk(
                user_preds, k=topk_max, sorted=True, largest=True
            )
            user_top_item_ids = user_top_item_ids.cpu().tolist()
            for top_k in top_k_list:
                if item_id in user_top_item_ids[:top_k]:
                    hit_rate[top_k].append(1)
                else:
                    hit_rate[top_k].append(0)

        precision_dict = {
            top_k: 100 * torch.mean(torch.tensor(hit_rate_list).float())
            for top_k, hit_rate_list in hit_rate.items()
        }

        for top_k, precision_val in precision_dict.items():
            logger.info(f"{top_k=} Precision={precision_val:.3f}%")
            tb_logger.experiment.add_scalar("Precision %", precision_val, top_k)
