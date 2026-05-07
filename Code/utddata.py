from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import _pickle as cPickle

from heterogeneity import dirichlet_split_non_iid, UTD_iid, UTDMHAD_modality_heterogeneity, shard_based_non_iid_split


def load_utd_pkl(path):
    with open(path, 'rb') as f:
        data = cPickle.load(f)

    return {
        "depth": data.depth,
        "inertial": data.inertial,
        "skeleton": data.skeleton,
        "labels": data.labels
    }


class UTDDataset(Dataset):

    def __init__(
            self,
            data_dict,
            normalize=True,
            owned_modalities=None,  # e.g. ['color','depth'] for this client; None means all
            missing_fill="zeros",  # "zeros" or "ones" or "none"
            device=None,  # optional torch device for placeholders
            dtype=torch.float32,
            label_dtype=torch.long,
            include_color=True,  # keep color in output even if model doesn't use it
            return_subnet_flag=True,
    ):
        self.depth = np.asarray(data_dict["depth"])
        self.inertial = np.asarray(data_dict["inertial"])
        self.skeleton = np.asarray(data_dict["skeleton"])
        self.labels = np.asarray(data_dict["labels"])

        self.dtype = dtype
        self.label_dtype = label_dtype
        self.include_color = include_color
        self.return_subnet_flag = return_subnet_flag

        # Owned modalities for this dataset/client
        self.owned = set(owned_modalities) if owned_modalities is not None else \
            {"depth", "inertial", "skeleton"}

        assert missing_fill in ["zeros", "ones", "none"]
        self.missing_fill = missing_fill
        self.device = device

        if normalize:
            self._normalize()

        # Precompute placeholder shapes using the first sample (assumes fixed shapes)
        self._shape_depth = self.depth[0].shape
        self._shape_inertial = self.inertial[0].shape
        self._shape_skeleton = self.skeleton[0].shape

    def _normalize(self):
        # NaN -> 0 (same spirit as your original)
        for arr_name in ["depth", "inertial", "skeleton"]:
            arr = getattr(self, arr_name)
            arr[arr != arr] = 0
            m = np.max(arr)
            if m != 0:
                setattr(self, arr_name, arr / m)

    def __len__(self):
        return len(self.labels)

    def _placeholder(self, shape):
        if self.missing_fill == "none":
            return None
        fill_value = 0.0 if self.missing_fill == "zeros" else 1.0
        t = torch.full(shape, fill_value, dtype=self.dtype)
        if self.device is not None:
            t = t.to(self.device)
        return t

    def __getitem__(self, idx):
        # For each modality: if owned -> real tensor; else -> placeholder/None

        if "depth" in self.owned:
            depth = torch.tensor(self.depth[idx], dtype=self.dtype)
        else:
            depth = self._placeholder(self._shape_depth)

        if "inertial" in self.owned:
            inertial = torch.tensor(self.inertial[idx], dtype=self.dtype)
        else:
            inertial = self._placeholder(self._shape_inertial)

        if "skeleton" in self.owned:
            skeleton = torch.tensor(self.skeleton[idx], dtype=self.dtype)
        else:
            skeleton = self._placeholder(self._shape_skeleton)

        label = torch.tensor(self.labels[idx], dtype=self.label_dtype).squeeze()

        # subnet_flag for your BlockTrainerBlend: [depth, inertial, skeleton]
        # (color is not part of subnet_flag in model.py)
        subnet_flag = torch.tensor(
            [
                1 if "depth" in self.owned else 0,
                1 if "inertial" in self.owned else 0,
                1 if "skeleton" in self.owned else 0,
            ],
            dtype=torch.long
        )

        # Build return tuple
        out = []
        if self.include_color:
            out.append(color)
        out.extend([depth, inertial, skeleton, label])

        if self.return_subnet_flag:
            out.append(subnet_flag)

        return tuple(out)


def build_federated_clients_pkl(
        args,
        train_pkl: str,
        test_pkl: str,
        num_clients: int,
        alpha: float,
        missing_prob: Dict[str, float],
        batch_size: int = 16,
        seed: int = 42,
        min_size: int = 1,
        ensure_at_least_one: bool = True,
) -> Tuple[Dict[int, DataLoader], DataLoader, Dict[int, List[str]], List[List[int]]]:
    """
    返回：
      client_loaders[cid]  -> DataLoader(client_dataset)
      test_loader          -> DataLoader
      owned[cid]           -> 该 client 拥有的模态列表
      client_indices       -> 该 client 拿到的样本索引列表
    """

    # 1) 读取训练集 pkl，并做一次全局归一化（强烈推荐：保证各 client 尺度一致）
    train_dict = load_utd_pkl(train_pkl)
    labels = np.asarray(train_dict["labels"])

    # 2) 数据异构（Dirichlet / IID）
    if args.data:
        client_indices = dirichlet_split_non_iid(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_size=min_size
        )
        # client_indices = shard_based_non_iid_split(
        #         labels=labels,
        #         num_clients=num_clients,
        #         num_shards=500,
        #         seed=seed,
        # )
    else:
        client_indices = UTD_iid(
            num_samples=len(labels),
            num_clients=num_clients,
            seed=seed,
            drop_remainder=True
        )

    # 3) 模态异构（按缺失概率）
    if args.modality:
        owned = UTDMHAD_modality_heterogeneity(
            num_clients=num_clients,
            missing_prob=missing_prob,
            ensure_at_least_one=ensure_at_least_one,
            seed=seed
        )
    else:
        owned = UTDMHAD_modality_heterogeneity(
            num_clients=num_clients,
            missing_prob={"depth": 0, "skeleton": 0, "inertial": 0},
            ensure_at_least_one=ensure_at_least_one,
            seed=seed
        )

    # 4) 为每个 client 构建自己的 Dataset（用 indices 切片 + owned_modalities 控制缺模态）
    client_loaders: Dict[int, DataLoader] = {}
    for cid in range(num_clients):
        idx = np.asarray(client_indices[cid], dtype=np.int64)

        data_cid = {
            "depth": train_dict["depth"][idx],
            "inertial": train_dict["inertial"][idx],
            "skeleton": train_dict["skeleton"][idx],
            "labels": train_dict["labels"][idx],
        }

        ds_cid = UTDDataset(
            data_dict=data_cid,
            normalize=True,
            owned_modalities=owned[cid],  # 用你的 heterogeneity 结果
            missing_fill="zeros",  # 推荐：保证默认 collate 不报错
            include_color=False,
            return_subnet_flag=True,
        )

        client_loaders[cid] = DataLoader(
            ds_cid,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    # 5) 测试集 loader：通常给全模态
    test_dict = load_utd_pkl(test_pkl)

    ds_test = UTDDataset(
        data_dict=test_dict,
        normalize=True,
        owned_modalities=["depth", "inertial", "skeleton"],
        missing_fill="zeros",
        include_color=False,
        return_subnet_flag=True,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return client_loaders, test_loader, owned, client_indices
