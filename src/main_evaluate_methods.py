import logging
import os
import os.path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset_utils import get_datasets
from lit_utils import LitModel

logger = logging.getLogger(__name__)


def mean_confidence_interval(data, confidence=0.9):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def compare_results(preds_dict: dict, eval_function, metric_dict: dict):
    for dataset_name, dataset_dict in preds_dict.items():
        print(dataset_name)
        if dataset_name in metric_dict:
            print(f"{dataset_name} exists in metric_dict")
            continue

        labels = dataset_dict["labels"]
        no_cf_preds = dataset_dict["no_cf_preds"]
        with_cf_preds = dataset_dict["with_cf_preds"]
        df = single_set_compare_results(
            labels, no_cf_preds, with_cf_preds, eval_function
        )
        metric_dict[dataset_name] = df
        print(df[["no_cf", "with_cf"]].round(2).T)
    return metric_dict


def single_label_ratio_compare_results(
    label_ratio, labels, preds_a, preds_b, eval_function
):
    # Define output
    res_dict = {"label_ratio": label_ratio}

    # Evaluate performance
    perf_a = eval_function(labels, preds_a)
    perf_b = eval_function(labels, preds_b)

    res_dict["pvalue"] = ttest_rel(perf_a, perf_b).pvalue

    # No CF
    res_dict["no_cf"] = np.mean(perf_a)
    res_dict["no_cf_std"] = np.std(perf_a)
    res_dict["no_cf_ci"] = mean_confidence_interval(perf_a)

    # With CF
    res_dict["with_cf"] = np.mean(perf_b)
    res_dict["with_cf_std"] = np.std(perf_b)
    res_dict["with_cf_ci"] = mean_confidence_interval(perf_b)

    return res_dict


def single_set_compare_results(
    labels, no_cf_pred_list, with_cf_pred_list, eval_function
):

    # Defining a dict
    res_dicts = []
    total = len(no_cf_pred_list)
    label_ratios = np.arange(0.1, 1.1, 0.1)
    for label_ratio, preds_a, preds_b in tqdm(
        zip(label_ratios, no_cf_pred_list, with_cf_pred_list), total=total
    ):

        res_dict = single_label_ratio_compare_results(
            label_ratio, labels, preds_a, preds_b, eval_function
        )
        res_dicts.append(res_dict)

    df = pd.DataFrame(res_dicts)
    df.set_index("label_ratio")
    df["improvement"] = df["with_cf"] / df["no_cf"] - 1.0

    return df


def plot_performance(df):
    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.plot(df.index, df["no_cf"], label="Baseline")
    ax.plot(df.index, df["with_cf"], label="With CF")

    for i, key in enumerate(["no_cf", "with_cf"]):
        ax.fill_between(
            df.index,
            df[key] - df[key + "_ci"],
            df[key] + df[key + "_ci"],
            color=f"C{i}",
            alpha=0.1,
        )

    ax.set_xlabel("Label ratio")
    ax.legend()
    return ax


def calc_top1_acc(labels, preds):
    return np.array(
        [labels[n][top1] for n, top1 in enumerate(np.argmax(preds, axis=1))]
    )


def calc_ap_score(labels, preds):
    aps = []
    num_experiments = 100
    num_samples = int(0.9 * len(labels))

    idxs_list = np.random.randint(
        low=0, high=len(labels), size=(num_experiments, num_samples)
    )
    for idxs in idxs_list:
        labels_chosen, preds_chosen = labels[idxs], preds[idxs]
        mask = labels_chosen.sum(axis=0) > 0
        ap = average_precision_score(labels_chosen[:, mask], preds_chosen[:, mask])
        aps.append(ap)
    return np.array(aps)


def cale_f1_score(labels, preds, thresh=0.95):
    f1s = []
    num_experiments = 10
    num_samples = int(0.9 * len(labels))
    for _ in range(num_experiments):
        idxs = np.random.randint(0, len(labels), num_samples)
        labels_chosen, preds_chosen = labels[idxs], preds[idxs]
        mask = labels_chosen.sum(axis=0) > 0
        f1 = f1_score(
            labels_chosen[:, mask], preds_chosen[:, mask] > thresh, average="samples"
        )
        f1s.append(f1)
    return np.array(f1s)


def plot_pvalues(df):
    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.plot(df.index, df["pvalue"])
    plt.axhline(y=0.05, color="r", linestyle="-", label="p=0.05")
    ax.set_xlabel("Label ratio")
    ax.set_ylabel("p-value")
    ax.legend()
    plt.show()


def build_label_ratio_dicts(results_path):
    res_dict = OmegaConf.load(results_path)

    # Build absolute path
    res_dict = {
        key: osp.join(res_dict["base_path"], value)
        for key, value in res_dict.items()
        if key != "base_path"
    }

    no_cf_dict = {key: value for key, value in res_dict.items() if "_no_cf" in key}
    with_cf_dict = {key: value for key, value in res_dict.items() if "_with_cf" in key}
    return no_cf_dict, with_cf_dict


def load_preds(base_path):
    no_cf_dict, with_cf_dict = build_label_ratio_dicts(base_path)

    labels = np.load(osp.join(list(no_cf_dict.values())[0], "labels.npy"))

    no_cf_preds, with_cf_preds = [], []
    for (key_a, path_a), (key_b, path_b) in zip(
        no_cf_dict.items(), with_cf_dict.items()
    ):
        preds_a = np.load(osp.join(path_a, "preds.npy"))
        preds_b = np.load(osp.join(path_b, "preds.npy"))

        ap_a = average_precision_score(labels, preds_a)  # ,average='micro')
        ap_b = average_precision_score(labels, preds_b)  # ,average='micro')

        print(f"{key_a} {key_b} [{ap_a:.3f} {ap_b:.3f}]. size={len(preds_a)}")
        no_cf_preds.append(preds_a)
        with_cf_preds.append(preds_b)

    return {
        "no_cf_preds": no_cf_preds,
        "with_cf_preds": with_cf_preds,
        "labels": labels,
    }


@hydra.main(
    config_path="../configs",
    config_name="evaluate_methods",
)
def evaluate_methods(cfg: DictConfig):
    os.chdir(get_original_cwd())

    plt.style.use(["science", "ieee"])
    out_path = osp.join("../outputs/figures")
    metric_res_dicts_path = osp.join(out_path, "metric_res_dicts.npy")

    dataset_mapping = {
        "Beauty": "Beauty",
        "Toys_and_Games": "Toys",
        "Clothing_Shoes_and_Jewelry": "Clothing",
        "movielens": "MovieLens",
    }

    preds_dict = {}
    for dataset_name, print_name in dataset_mapping.items():
        print(dataset_name)
        preds_dict[print_name] = load_preds(
            osp.join(f"../outputs/{dataset_name}/results.yaml")
        )

    metric_funcs = {
        "ap": calc_ap_score,
        "top1_acc": calc_top1_acc,
    }  # , 'f1':cale_f1_score}

    if osp.exists(metric_res_dicts_path):
        metric_res_dicts = np.load(metric_res_dicts_path, allow_pickle=True).item()
    else:
        metric_res_dicts = {}

    for metric_name, metric_func in metric_funcs.items():
        logger.info(metric_name)

        # Initilize output: if metric exsits, use previous results
        single_metric_res_dict = {}
        if metric_name in metric_res_dicts:
            single_metric_res_dict = metric_res_dicts[metric_name]

        # metric -> dataset -> performance dataframe
        single_metric_res_dict = compare_results(
            preds_dict, metric_func, single_metric_res_dict
        )

        # Add to dict
        metric_res_dicts[metric_name] = single_metric_res_dict
        np.save(metric_res_dicts_path, metric_res_dicts)
        print()

    np.save(metric_res_dicts_path, metric_res_dicts)


if __name__ == "__main__":
    evaluate_methods()
