import numpy as np
import torch
from typing import Dict, List, Iterable, Tuple, Optional, Any


# 1. device heterogeneity
def generate_heterogeneous(
    var: float,
    mean: float,
    num: int,
    g: torch.Generator
):
    std = float(np.sqrt(var))
    samples = []

    while len(samples) < num:
        s = torch.normal(mean, std, size=(num,), generator=g)
        s = s[(s >= 0.0) & (s <= 1.0)]
        samples.extend(s.tolist())

    return samples[:num]


def inverse_minmax(x_norm, data_min, data_max, feature_range=(0.0, 1.0)):
    x_norm = np.asarray(x_norm, dtype=float)
    data_min = np.asarray(data_min, dtype=float)
    data_max = np.asarray(data_max, dtype=float)
    a, b = feature_range

    scale = (data_max - data_min)
    zero_mask = (scale == 0)
    x = ((x_norm - a) / (b - a)) * scale + data_min

    if np.any(zero_mask):
        x = np.where(zero_mask, data_min, x)
    return x


def heterogeneous_matrix(args):
    g = torch.Generator().manual_seed(args.seed)
    if args.num_edges > args.num_clients:
        print("n must not be less than e. Please set the parameter again.")
        return None

    if not args.device:
        matrix_comm = np.full((args.num_clients, args.num_edges), 30.0)
        matrix_comp = np.full((args.num_clients,), 0.5)
    

    else:
        matrix_comm = [generate_heterogeneous(args.var, args.mean, args.num_clients, g) for _ in range(args.num_edges)]
        matrix_comm = np.array(matrix_comm)
        matrix_comm = inverse_minmax(matrix_comm, 0, 30, (0, 1))
        # matrix_comm = inverse_minmax(matrix_comm, 0.5, 1.0, (0, 1))
        matrix_comm = matrix_comm.T

        matrix_comp = generate_heterogeneous(args.var, args.mean, args.num_clients, g)
        matrix_comp = np.array(matrix_comp)
        # matrix_comp = inverse_minmax(matrix_comp, 0, 1, (0, 1))
        # matrix_comp = inverse_minmax(matrix_comp, 0.06, 0.024, (0, 1))
    
    matrix_heterogeneity = np.round(matrix_comm + matrix_comp[:, None],3)

    # matrix_heterogeneity = matrix_comm + matrix_comp[:, np.newaxis]
    # matrix_heterogeneity = np.round(np.array(matrix_heterogeneity), 3)
    return matrix_heterogeneity, matrix_comp


# 2. data heterogeneity

def shard_based_non_iid_split(
    labels: np.ndarray,
    num_clients: int,
    num_shards: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    基于 shard 的可控 non-iid 划分方法

    参数:
        labels: (N,)
        num_clients: client 数
        num_shards: shard 总数 (控制 non-iid 强度)

    返回:
        client_idx[k] = 第k个client的样本索引列表
    """

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    N = len(labels)

    if num_shards < num_clients:
        raise ValueError("num_shards must be >= num_clients")

    # -------- Step 1: 按 label 排序 --------
    idxs = np.argsort(labels)

    # -------- Step 2: 切成 shards --------
    shard_size = N // num_shards
    usable_len = shard_size * num_shards
    idxs = idxs[:usable_len]

    shards = np.split(idxs, num_shards)

    # -------- Step 3: 随机打乱 shard 顺序 --------
    rng.shuffle(shards)

    # -------- Step 4: 分配给 clients --------
    shards_per_client = num_shards // num_clients

    client_idx = []
    for i in range(num_clients):
        assigned_shards = shards[
            i * shards_per_client : (i + 1) * shards_per_client
        ]
        client_data = np.concatenate(assigned_shards)
        rng.shuffle(client_data)
        client_idx.append(client_data.tolist())

    return client_idx

def dirichlet_split_non_iid(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_size: int = 1,
) -> List[List[int]]:
    """
    labels: shape (N,)
    返回:
        client_idx[k] = 第k个client对应的样本索引list
    """

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)

    if n < num_clients * min_size:
        raise ValueError(
            f"Total samples {n} < num_clients*min_size {num_clients*min_size}"
        )

    classes = np.unique(labels)
    class_indices = [np.where(labels == c)[0] for c in classes]

    # -------- Step 1: Dirichlet 按类别分配 --------
    buckets = [[] for _ in range(num_clients)]
    sizes = np.zeros(num_clients, dtype=np.int64)

    for idx_c in class_indices:
        idx_c = idx_c.copy()
        rng.shuffle(idx_c)

        p = rng.dirichlet(np.full(num_clients, alpha))
        cnt = rng.multinomial(len(idx_c), p)

        off = 0
        for k in range(num_clients):
            ck = cnt[k]
            if ck > 0:
                buckets[k].append(idx_c[off:off + ck])
                sizes[k] += ck
                off += ck

    # -------- Step 2: 拼接成数组 --------
    client_arr = []
    for k in range(num_clients):
        arr = (
            np.concatenate(buckets[k])
            if buckets[k]
            else np.empty(0, dtype=np.int64)
        )
        rng.shuffle(arr)
        client_arr.append(arr)

    # -------- Step 3: 修复 min_size --------
    current_sizes = np.array([len(a) for a in client_arr])
    need = np.where(current_sizes < min_size)[0].tolist()
    rich = np.where(current_sizes > min_size)[0].tolist()

    while need:
        i = need.pop()
        deficit = min_size - len(client_arr[i])
        if deficit <= 0:
            continue

        if not rich:
            rich = np.where(
                np.array([len(a) for a in client_arr]) > min_size
            )[0].tolist()
            if not rich:
                raise RuntimeError("Cannot satisfy min_size constraint.")

        d = rich[rng.integers(0, len(rich))]
        donor = client_arr[d]
        donor_extra = len(donor) - min_size
        take = min(deficit, donor_extra)

        give = donor[-take:]
        client_arr[d] = donor[:-take]
        client_arr[i] = np.concatenate([client_arr[i], give])

        if len(client_arr[d]) <= min_size:
            rich = [x for x in rich if x != d]
        if len(client_arr[i]) < min_size:
            need.append(i)

    client_idx = [arr.astype(int).tolist() for arr in client_arr]

    return client_idx

# def dirichlet_split_non_iid(
#         labels: np.ndarray,
#         num_clients: int,
#         alpha: float,
#         seed: int = 42,
#         min_size: int = 10,
# ) -> List[List[int]]:
#     """
#     labels: shape (N,)
#     返回: client_indices[k] = 该 client 拿到的样本索引列表
#     """
#     rng = np.random.default_rng(seed)
#     labels = np.asarray(labels)
#     classes = np.unique(labels)
#     class_indices = {c: np.where(labels == c)[0] for c in classes}
#
#     while True:
#         client_idx = [[] for _ in range(num_clients)]
#
#         for c in classes:
#             idx_c = class_indices[c].copy()
#             rng.shuffle(idx_c)
#
#             proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
#             split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
#             splits = np.split(idx_c, split_points)
#
#             for k in range(num_clients):
#                 client_idx[k].extend(splits[k].tolist())
#
#         sizes = [len(x) for x in client_idx]
#         if min(sizes) >= min_size:
#             for k in range(num_clients):
#                 rng.shuffle(client_idx[k])
#             return client_idx


def UTD_iid(
        num_samples: int,
        num_clients: int,
        seed: int = 42,
        drop_remainder: bool = True
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(num_samples)
    rng.shuffle(idxs)

    if drop_remainder:
        n = (num_samples // num_clients) * num_clients
        idxs = idxs[:n]
        client_splits = np.array_split(idxs, num_clients)
        return [s.tolist() for s in client_splits]
    else:
        base = num_samples // num_clients
        r = num_samples % num_clients
        client_idx = []
        start = 0
        for cid in range(num_clients):
            size = base + (1 if cid < r else 0)
            client_idx.append(idxs[start:start + size].tolist())
            start += size
        return client_idx


# 3. modality heterogeneity
def UTDMHAD_modality_heterogeneity(
        num_clients: int,
        missing_prob: Dict[str, float],
        ensure_at_least_one: bool = True,
        modalities=None,
        seed: int = 42
) -> Dict[int, List[str]]:
    """
    missing_prob[m] = P(client 缺失模态 m)
    返回 owned[cid] = client 拥有的模态列表
    """
    if modalities is None:
        modalities = ['depth', 'skeleton', 'inertial']
    rng = np.random.default_rng(seed)

    owned: Dict[int, List[str]] = {}
    for cid in range(num_clients):
        while True:
            mods = []
            for m in modalities:
                p_miss = float(missing_prob.get(m, 0.5))
                missing = (rng.random() < p_miss)
                if not missing:
                    mods.append(m)

            if mods or not ensure_at_least_one:
                if not mods:
                    m_keep = min(missing_prob, key=missing_prob.get)
                    mods = [m_keep]
                owned[cid] = mods
                break

    return owned
