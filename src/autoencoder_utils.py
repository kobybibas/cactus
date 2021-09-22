import logging
import os
import os.path as osp
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from src.manifold_utils import read_data_from_manifold
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AutoEncoderTrainset(data.Dataset):
    def __init__(self, user_mat: torch.Tensor, num_users: int, num_items: int):

        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_mat = user_mat

    def __len__(self):
        return len(self.user_mat)

    def __getitem__(self, idx):
        # Sparse vec: 1 for item with intercation and 0 for non intercation
        user_vec = self.user_mat[idx]
        return user_vec


class AutoEncoderPairsSet(data.Dataset):
    def __init__(self, user_mat: torch.Tensor, test_set_pairs: torch.Tensor):
        super().__init__()
        self.user_mat = user_mat
        self.test_set_pairs = test_set_pairs

    def __len__(self):
        return len(self.test_set_pairs)

    def __getitem__(self, idx):
        # Sparse vec: 1 for item with intercation and 0 for non intercationץ
        # user mat is set to 0 for item_id
        user_id, item_id = self.test_set_pairs[idx]
        user_vec = torch.clone(self.user_mat[user_id])
        user_vec[item_id] = 0
        return user_vec, item_id


def get_autoencoder_datasets(data_dir: str, category: str, local_dir_path: str):
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

    train_set_pairs, test_set_pairs = pkl_data["train_set"], pkl_data["test_set"]
    num_users, num_items = pkl_data["user_count"], pkl_data["item_count"]
    logger.info(f"{num_users=} {num_items=}")

    # Group by user_id
    user_item_list = np.split(
        train_set_pairs[:, 1],
        np.unique(train_set_pairs[:, 0], return_index=True)[1][1:],
    )

    # Build dataset
    user_matrix = torch.zeros(num_users, num_items)
    for user_id, user_items in enumerate(user_item_list):
        user_matrix[user_id, user_items] = 1

    train_dataset = AutoEncoderTrainset(torch.clone(user_matrix), num_users, num_items)

    # Build pairs set
    train_set_pairs = torch.from_numpy(train_set_pairs)
    test_set_pairs = torch.from_numpy(test_set_pairs)  # (user_id, item_id)
    train_pairs_dataset = AutoEncoderPairsSet(torch.clone(user_matrix), train_set_pairs)
    test_pairs_dataset = AutoEncoderPairsSet(torch.clone(user_matrix), test_set_pairs)
    return train_dataset, train_pairs_dataset, test_pairs_dataset


class AutoEncoderModel(nn.Module):
    def __init__(self, num_items: int, cf_size: int, drop_out_p: float = 0.1):
        super().__init__()

        self.item_embs = nn.Linear(num_items, cf_size)  # Last dim is the bias
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.item_bias.weight, std=0.01)

        self.bottelneck = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cf_size, 16),
            nn.Dropout(p=drop_out_p),
            nn.ReLU(),
            nn.Linear(16, 8),  # bottelneck
            nn.Dropout(p=drop_out_p),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.Dropout(p=drop_out_p),
            nn.ReLU(),
            nn.Linear(16, cf_size, bias=False),
            nn.ReLU(),
        )
        embed_items, embed_items_bias = self.get_embedding_weights()
        logger.info(
            f"AutoEncoderModel __init__: {embed_items.shape=} {embed_items_bias.shape=}"
        )

    def forward(self, x):
        z = self.item_embs(x)
        z = self.bottelneck(z)
        y = nn.functional.linear(z, self.item_embs.weight.t())
        return (
            y + self.item_bias.weight.t()
        )  # y.shape=[batch, num_items]. item_bias.weight.t().shape=[1, num_items]

    def get_embedding_weights(self):
        return self.item_embs.weight.t(), self.item_bias.weight


class AutoEncoderLit(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, user_vec):
        return self.model(user_vec)

    def calc_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, gt, reduction="none"
        )

        # Construct weight
        weight = torch.ones_like(gt)
        weight[gt == 0] = self.cfg["weight_neg_items"]
        loss = (weight * loss).sum(axis=-1).mean()
        return loss

    def calc_acc(self, pos, neg):
        return (pos > neg).float().mean()

    def training_step(self, batch, batch_idx):
        user_vec_gt = batch  # sparse vector: 1 for item with intercation and 0 for non interaction

        # Predict positive item
        user_vec_pred = self(user_vec_gt)

        # Metrics
        loss = self.calc_loss(user_vec_pred, user_vec_gt)

        # Log
        phase = "train"
        self.log_dict(
            {f"loss/{phase}": loss},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return {"loss": loss}

    def epoch_end_helper(self, outputs, phase: str):
        assert phase in ["train", "val"]
        loss = torch.mean(torch.tensor([out["loss"] for out in outputs])).item()
        acc = torch.mean(torch.tensor([out["acc"] for out in outputs])).item()
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] {phase} epoch end. [Loss Acc]=[{loss:.4f} {acc:.2f}]"
        )

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor([out["loss"] for out in outputs])).item()
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] Train epoch end. loss={loss:.4f}"
        )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        (
            user_vec_gt,
            pos_id,
        ) = batch  # user_vec_gt is a parse vector: 1 for item with intercation 0 otherwise

        # Predict positive item. user_vec_gt is set to 0 for pos_id
        user_vec_pred = self(user_vec_gt)

        # Random sample from positive
        user_vec_pred_cpu = user_vec_pred.detach().cpu()
        pos_idx = pos_id.unsqueeze(0).long().to(user_vec_pred_cpu.device)
        pos_score = user_vec_pred_cpu.gather(1, pos_idx.view(-1, 1))

        # Random sample from negative
        batch_size, num_items = user_vec_pred.size(0), user_vec_pred.size(1)
        neg_idx = torch.randint(num_items, (batch_size, 1)).to(user_vec_pred_cpu.device)
        neg_score = user_vec_pred_cpu.gather(1, neg_idx.view(-1, 1))

        # Metrics
        # user_vec_gt[pos_idx.long()] = 1
        loss = torch.tensor([0]).float()  # self.calc_loss(user_vec_pred, user_vec_gt)
        acc = self.calc_acc(pos_score, neg_score)

        # Log
        self.log_dict(
            {f"loss/val_{dataloader_idx}": loss, f"acc/val_{dataloader_idx}": acc},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs_all_loaders):
        for i, outputs in enumerate(outputs_all_loaders):
            loss = torch.mean(torch.tensor([out["loss"] for out in outputs])).item()
            acc = torch.mean(torch.tensor([out["acc"] for out in outputs])).item()
            logger.info(
                f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] [{i}] Val epoch end. loss={loss:.4f} acc={acc:.4f}"
            )

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     self.cfg["lr"],
        #     weight_decay=self.cfg["weight_decay"],
        # )
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg["milestones"]
        )
        return [optimizer], [lr_scheduler]

    def calc_topk(self, dataset, top_k_list, tb_logger=None):
        # Evaluate precision for various k
        hit_rate = {top_k: [] for top_k in top_k_list}
        topk_max = torch.max(torch.tensor(top_k_list))
        for user_vec_gt, pos_id in tqdm(dataset):
            # Predict recmmended items. Notice user_vec_pred is 0 for pos_id.
            user_vec_pred = self.model(user_vec_gt).squeeze()

            # Ignore training items
            user_vec_pred[user_vec_gt == 1] = user_vec_pred.min()

            # Get top k
            _, user_top_item_ids = torch.topk(
                user_vec_pred, k=topk_max, sorted=True, largest=True
            )
            user_top_item_ids = user_top_item_ids.cpu().tolist()

            # Check if item is in topk
            for top_k in top_k_list:
                if pos_id in user_top_item_ids[:top_k]:
                    hit_rate[top_k].append(1)
                else:
                    hit_rate[top_k].append(0)

        # Calculate precision
        precision_dict = {
            top_k: 100 * torch.mean(torch.tensor(hit_rate_list).float())
            for top_k, hit_rate_list in hit_rate.items()
        }

        # Log precision
        for top_k, precision_val in precision_dict.items():
            logger.info(f"{top_k=} {precision_val=}")
            if tb_logger is not None:
                tb_logger.experiment.add_scalar("Precision %", precision_val, top_k)
