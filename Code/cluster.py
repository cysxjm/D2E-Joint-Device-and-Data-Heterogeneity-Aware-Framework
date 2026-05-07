from os import times

import numpy as np
import random
import torch

from scipy.spatial.distance import cosine
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional



def cluster(times, client):
    sorted_ti = []
    sorted_cl = []

    for t_row, c_row in zip(times, client):
        indices = sorted(range(len(t_row)), key=lambda i: t_row[i])
        sorted_t_row = [t_row[i] for i in indices]
        sorted_c_row = [c_row[i] for i in indices]
        sorted_ti.append(sorted_t_row)
        sorted_cl.append(sorted_c_row)

    max_values = [max(sublist) for sublist in sorted_ti]
    original_indices = list(range(len(sorted_cl)))
    sorted_pairs = sorted(zip(max_values, sorted_ti, sorted_cl, original_indices), key=lambda x: x[0])

    sorted_times = [pair[1] for pair in sorted_pairs]
    sorted_client = [pair[2] for pair in sorted_pairs]
    sorted_edge = [pair[3] for pair in sorted_pairs]

    return sorted_times, sorted_client, sorted_edge


def hierfavg_cluster(matrix):
    N, E = matrix.shape
    selected_elements = []
    selected_rows = set()
    column_count = {i: 0 for i in range(E)}
    max_per_column = N // E

    elements = [(matrix[i, j], i, j) for i in range(N) for j in range(E)]
    seed = 42
    rng = random.Random(seed)
    rng.shuffle(elements)

    for value, row, col in elements:
        if row not in selected_rows and column_count[col] < max_per_column:
            selected_elements.append((value, row, col))
            selected_rows.add(row)
            column_count[col] += 1
        if len(selected_elements) == N:
            break
    times_by_col = [[] for _ in range(E)]
    client_by_col = [[] for _ in range(E)]
    for val, row, col in selected_elements:
        times_by_col[col].append(val)
        client_by_col[col].append(row)

    sorted_times, sorted_client, sorted_edge = cluster(times_by_col, client_by_col)
    return sorted_times, sorted_client, sorted_edge


def fedfleet_cluster(matrix):
    selected = select_elements(matrix)
    sorted_times = [[] for _ in range(matrix.shape[1])]
    sorted_client = [[] for _ in range(matrix.shape[1])]
    for val, row, col in selected:
        sorted_times[col].append(val)
        sorted_client[col].append(row)
    sorted_times, sorted_client, sorted_edge = cluster(sorted_times, sorted_client)
    return sorted_times, sorted_client, sorted_edge


def safi_cluster(matrix, comp_time):
    N, E = matrix.shape
    client_size = N // E
    indices = sort_comp(comp_time)

    sorted_client = [[] for _ in range(E)]
    sorted_times = [[] for _ in range(E)]

    for e in range(E):
        for i in range(client_size):
            sorted_client[e].append(indices[e * client_size + i])
            sorted_times[e].append(matrix[indices[e * client_size + i]][e])

    sorted_times, sorted_client, sorted_edge = cluster(sorted_times, sorted_client)
    return sorted_times, sorted_client, sorted_edge


def _state_dicts_to_cosine_sim(model_states):
    """
    model_states: list of state_dict (tensor on cpu)
    return: S (N,N) cosine similarity matrix in [ -1, 1 ]
    """
    vecs = []
    for sd in model_states:
        # 保证顺序一致：按 key 排序拼接
        flat = []
        for k in sorted(sd.keys()):
            v = sd[k]
            if isinstance(v, torch.Tensor):
                flat.append(v.detach().reshape(-1).float().cpu())
        if len(flat) == 0:
            raise ValueError("Empty state_dict (no tensor params).")
        vec = torch.cat(flat)
        vecs.append(vec)

    X = torch.stack(vecs, dim=0)  # (N, D)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    S = (X @ X.t()).numpy()  # (N, N)
    return S


def X_cluster(matrix, model, alpha):
    T = np.asarray(matrix, float)

    # ---- FIX START ----
    # model 既可以是 NxN 的相似度矩阵，也可以是 list[state_dict]
    if isinstance(model, (list, tuple)) and len(model) > 0 and isinstance(model[0], dict):
        S = _state_dicts_to_cosine_sim(model)  # (N, N)
    else:
        S = np.asarray(model, float)  # 允许你直接传 NxN
    # ---- FIX END ----

    N, E = T.shape
    if S.shape != (N, N):
        raise ValueError(f"S must be shape (N,N) == ({N},{N}), got {S.shape}")

    order = np.argsort(-T.mean(1))  # desc avg latency
    clients = [[] for _ in range(E)]
    n = np.zeros(E, int)
    s = np.zeros(E, float)
    ssq = np.zeros(E, float)
    pair = np.zeros(E, float)  # sum_{i<j in edge} (1 - S_ij)
    assigned = np.zeros(N, bool)

    def std(nn, ss, sssq):
        if nn <= 1: return 0.0
        m = ss / nn
        v = (sssq / nn) - m * m
        return float(np.sqrt(v if v > 0 else 0.0))

    def div(nn, p):
        if nn <= 1: return 0.0
        return float(p / (nn * (nn - 1) / 2.0))

    def norm(x):
        a, b = float(x.min()), float(x.max())
        return np.zeros_like(x) if b - a < 1e-12 else (x - a) / (b - a + 1e-12)

    # seed: one client per edge
    p = 0
    for e in range(E):
        c = int(order[p])
        p += 1
        assigned[c] = True
        clients[e].append(c)
        n[e] = 1
        s[e] = T[c, e]
        ssq[e] = T[c, e] * T[c, e]

    # greedy
    for c in order:
        c = int(c)
        if assigned[c]:
            continue

        ddev = np.zeros(E)
        ddata = np.zeros(E)

        for e in range(E):
            old_std = std(n[e], s[e], ssq[e])
            t = T[c, e]
            new_std = std(n[e] + 1, s[e] + t, ssq[e] + t * t)
            ddev[e] = new_std - old_std

            old_div = div(n[e], pair[e])
            add = 0.0 if n[e] == 0 else float(np.sum(1.0 - S[np.array(clients[e], int), c]))
            new_div = div(n[e] + 1, pair[e] + add)
            ddata[e] = new_div - old_div

        score = alpha * norm(ddev) - (1.0 - alpha) * norm(ddata)
        e_star = int(np.argmin(score))

        add = 0.0 if n[e_star] == 0 else float(np.sum(1.0 - S[np.array(clients[e_star], int), c]))
        clients[e_star].append(c)
        pair[e_star] += add
        t = T[c, e_star]
        n[e_star] += 1
        s[e_star] += t
        ssq[e_star] += t * t
        assigned[c] = True

    times = [[float(T[c, e]) for c in clients[e]] for e in range(E)]
    sorted_times, sorted_client, sorted_edge = cluster(times, clients)
    return sorted_times, sorted_client, sorted_edge

# def X_cluster(matrix, model, alpha):
    """
    Algorithm 3: Joint Device-and-Data Aware Client-to-Edge Association

    Returns (aligned with X_cluster):
        sorted_times
        sorted_client
        sorted_edge
    """

    # ---- Parse inputs ----
    T = np.asarray(matrix, float)

    # model 可以是 NxN 相似度矩阵，也可以是 state_dict list
    if isinstance(model, (list, tuple)) and len(model) > 0 and isinstance(model[0], dict):
        S = _state_dicts_to_cosine_sim(model)
    else:
        S = np.asarray(model, float)

    N, E = T.shape
    if S.shape != (N, N):
        raise ValueError(f"S must be shape (N,N) == ({N},{N}), got {S.shape}")

    # ---- Helper functions ----
    def std_latency(edge, assigned):
        if len(assigned) <= 1:
            return 0.0
        vals = T[np.array(assigned, int), edge]
        return float(np.std(vals))

    def diversity(assigned):
        k = len(assigned)
        if k <= 1:
            return 0.0
        idx = np.array(assigned, int)
        subS = S[np.ix_(idx, idx)]
        triu = np.triu_indices(k, 1)
        pair_sum = float(np.sum(1.0 - subS[triu]))
        return pair_sum / (k * k)

    def minmax_normalize(x):
        xmin, xmax = float(x.min()), float(x.max())
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    # ---- Initialization ----
    t_tilde = T.mean(axis=1)
    order = np.argsort(-t_tilde)

    clients = [[] for _ in range(E)]
    assigned = np.zeros(N, bool)

    # seed: one client per edge
    p = 0
    for e in range(E):
        c = int(order[p])
        p += 1
        clients[e].append(c)
        assigned[c] = True

    # ---- Greedy assignment ----
    for c in order:
        c = int(c)
        if assigned[c]:
            continue

        delta_dev = np.zeros(E)
        delta_data = np.zeros(E)

        for e in range(E):
            Ce = clients[e]

            dev_before = std_latency(e, Ce)
            dev_after = std_latency(e, Ce + [c])
            delta_dev[e] = dev_after - dev_before

            data_before = diversity(Ce)
            data_after = diversity(Ce + [c])
            delta_data[e] = data_after - data_before

        delta_dev_hat = minmax_normalize(delta_dev)
        delta_data_hat = minmax_normalize(delta_data)

        score = alpha * delta_dev_hat - (1.0 - alpha) * delta_data_hat
        e_star = int(np.argmin(score))

        clients[e_star].append(c)
        assigned[c] = True

    # ---- Build times ----
    times = [[float(T[c, e]) for c in clients[e]] for e in range(E)]

    # ---- 与你原函数保持一致的返回格式 ----
    sorted_times, sorted_client, sorted_edge = cluster(times, clients)

    return sorted_times, sorted_client, sorted_edge

def sort_comp(comp_time):
    sorted_indices = sorted(range(len(comp_time)), key=lambda i: comp_time[i])
    return sorted_indices


def select_elements(matrix):
    N, E = matrix.shape
    selected_elements = []
    selected_rows = set()
    column_count = {i: 0 for i in range(E)}
    max_per_column = N // E

    elements = [(matrix[i, j], i, j) for i in range(N) for j in range(E)]
    elements.sort()

    for value, row, col in elements:
        if row not in selected_rows and column_count[col] < max_per_column:
            selected_elements.append((value, row, col))
            selected_rows.add(row)
            column_count[col] += 1
        if len(selected_elements) == N:
            break

    return selected_elements


def GCC(models, fixed_indices, constraints, num_groups):
    indices = set(range(len(models)))
    remaining_indices = list(indices - set(fixed_indices))
    group_size = len(models) // num_groups

    distance_matrix = calculate_distance_matrix(models)

    groups = [[] for _ in range(num_groups)]
    for i, fixed_index in enumerate(fixed_indices):
        groups[i].append(fixed_index)

    mean_distances = {
        idx: np.mean(distance_matrix[idx]) for idx in remaining_indices
    }
    remaining_indices.sort(key=lambda idx: mean_distances[idx], reverse=True)

    for idx in remaining_indices:
        best_group = None
        max_diversity_gain = -float('inf')

        for i in range(num_groups):
            if len(groups[i]) < group_size and idx not in set(constraints[i]):
                new_group = groups[i] + [idx]
                diversity_gain = evaluate_partition(distance_matrix, new_group) - evaluate_partition(distance_matrix,
                                                                                                     groups[i])
                if diversity_gain > max_diversity_gain:
                    max_diversity_gain = diversity_gain
                    best_group = i

        if best_group is not None:
            groups[best_group].append(idx)

    return groups


def flatten_model_parameters(state_dict):
    return np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])


def calculate_distance_matrix(models):
    num_models = len(models)
    distance_matrix = np.zeros((num_models, num_models))
    for i in range(num_models):
        vec_i = flatten_model_parameters(models[i])
        for j in range(i + 1, num_models):
            vec_j = flatten_model_parameters(models[j])
            dist = cosine(vec_i, vec_j)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def evaluate_partition(distance_matrix, group):
    if len(group) < 2:
        return 0
    total = 0
    count = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            total += distance_matrix[group[i]][group[j]]
            count += 1
    return total / count


def SSE(matrix):
    N = len(matrix)
    if N == 0:
        return None, []
    E = len(matrix[0])
    if E == 0:
        return None, []

    col_candidates = []

    queue_values = [0] * E
    queue_rows = [0] * E

    for j in range(E):
        col_vals = [(matrix[i][j], i) for i in range(N)]
        col_vals.sort(key=lambda x: x[0], reverse=True)
        queue_values[j] = col_vals[0][0]
        queue_rows[j] = col_vals[0][1]
        col_candidates.append(col_vals)

    pointers = [1] * E

    replace_counts = [1] * E

    replaced_rows_by_col = [[] for _ in range(E)]

    states = []

    def record_state():
        diff = max(queue_values) - min(queue_values)
        col_repl_copy = [col_list[:] for col_list in replaced_rows_by_col]
        states.append((diff, queue_rows[:], col_repl_copy))

    record_state()

    def check_duplicate_stop():
        row_count = {}
        for col_list in replaced_rows_by_col:
            for row in col_list:
                row_count[row] = row_count.get(row, 0) + 1
                if row_count[row] >= E:
                    return True
        return False

    while True:
        if any(count > N - (N // E) for count in replace_counts) or check_duplicate_stop():
            break
        current_max = max(queue_values)
        col_of_max = queue_values.index(current_max)

        idx = pointers[col_of_max]
        if idx >= len(col_candidates[col_of_max]):
            break

        old_row = queue_rows[col_of_max]
        replaced_rows_by_col[col_of_max].append(old_row)

        val, row = col_candidates[col_of_max][idx]
        pointers[col_of_max] += 1
        replace_counts[col_of_max] += 1

        queue_values[col_of_max] = val
        queue_rows[col_of_max] = row

        record_state()

    states.sort(key=lambda x: x[0])
    for diff, rows, col_replacements in states:
        if len(set(rows)) < E:
            continue
        return rows, col_replacements

    return None, []
