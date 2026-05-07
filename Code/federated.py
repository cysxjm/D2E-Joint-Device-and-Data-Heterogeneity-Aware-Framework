import copy
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cluster import X_cluster, fedfleet_cluster, hierfavg_cluster, safi_cluster
from train import train, evaluate


def Sync_agg(local_states, num_samples):
    avg = {k: v.detach().cpu().clone() for k, v in local_states[0].items()}
    with torch.no_grad():
        for k in avg.keys():
            if torch.is_floating_point(avg[k]):
                avg[k].zero_()
                for local_state, num_sample in zip(local_states, num_samples):
                    avg[k].add_(local_state[k].detach().cpu(), alpha=num_sample / (sum(num_samples)))
            else:
                avg[k].copy_(local_states[0][k].detach().cpu())
    return avg


def Staleness_hinge(s, a=1.0, b=3.0):
    # s: non-negative staleness
    if s <= b:
        return 1.0
    return 1.0 / (a * (s - b) + 1.0)

def Async_agg(w_client, w_global, t, tau, alpha, n_i, N, clip=True):
    # staleness should be >= 0
    s = max(0.0, float(t) - float(tau))
    decay = Staleness_hinge(s)

    sample_weight = float(n_i) / float(N)
    a_t = float(alpha) * float(decay) * float(sample_weight)

    if clip:
        a_t = max(0.0, min(1.0, a_t))

    new_w = {}
    with torch.no_grad():
        for k in w_global.keys():
            w_g = w_global[k]
            if torch.is_floating_point(w_g):
                w_c = w_client[k].to(device=w_g.device, dtype=w_g.dtype)
                new_w[k] = (1.0 - a_t) * w_g + a_t * w_c
            else:
                new_w[k] = w_g.clone()
    return new_w


def Sync_Sync(sorted_times, sorted_client, sorted_edge, local_states, num_samples):
    edge_client_states, edge_samples, edge_states = [], [], []
    max_t = max(sorted_times[-1])
    for num_e in range(len(sorted_edge)):
        edge_client_states = []
        edge_samples = []

        edge_client_idx = sorted_client[num_e]
        for client in edge_client_idx:
            edge_client_states.append(local_states[client])
            edge_samples.append(num_samples[client])

        edge_states.append(Sync_agg(edge_client_states, edge_samples))
        time.sleep(max_t)

    edge_weights = [sum(sorted_edge_samples) for sorted_edge_samples in
                    [[num_samples[c] for c in sorted_client[e]] for e in range(len(sorted_edge))]]
    global_state = Sync_agg(edge_states, edge_weights)

    return global_state


def Sync_Async(sorted_times, sorted_client, sorted_edge, local_states, num_samples):
    edge_client_states, edge_samples, edge_states = [], [], []
    num_e, num_c, t, tao = 0, 0, 0, 0
    total_sample = sum(num_samples)
    for num_e in range(len(sorted_edge)):
        edge_client_idx = sorted_client[num_e]
        for client in edge_client_idx:
            edge_client_states.append(local_states[client])
            edge_samples.append(num_samples[client])
            num_c += 1
        edge_states.append(Sync_agg(edge_client_states, edge_samples))
        t = sorted_times[num_e][num_c - 1]
        time.sleep(t)
        edge_sample = sum(edge_samples)
        alpha = 1 - (num_e / len(sorted_edge))
        if num_e == 0:
            global_state = edge_states[num_e]
        else:
            global_state = Async_agg(edge_states[num_e], global_state, t, tao, alpha, edge_sample, total_sample)
        tao = t
        num_c = 0
        edge_client_states, edge_samples = [], []
    return global_state


def get_base_logdir(args) -> str:
    if args.device and not args.data and not args.modality:
        base_logdir = f"{args.a}_device_logs/{args.a}_{args.var}_device"
    elif not args.device and args.data and not args.modality:
        base_logdir = f"{args.a}_data_logs/{args.a}_{args.alpha_de}_data"
    elif not args.device and not args.data and args.modality:
        base_logdir = f"{args.a}_modality_logs/{args.ownership_prob}_modality"
    else:
        base_logdir = (
            f"senstive/{args.a}/{args.device}{args.var}_device_"
            f"{args.data}{args.alpha_de}_data_"
            f"{args.modality}{args.ownership_prob}_modality"
        )
    os.makedirs(base_logdir, exist_ok=True)
    return base_logdir


def log_eval(writer, prefix: str, val_loss: float, val_acc: float, step: int):
    writer.add_scalar(f"{prefix}/Loss", val_loss, step)
    writer.add_scalar(f"{prefix}/Accuracy", val_acc, step)


def build_cfg_for_fl(
        fl_name: str,
        args,
        global_model_template,
        train_loaders,
        heterogeneity,
        comp,
        device,
        criterion,
        base_logdir: str,
):
    """
    依据 fl_name 构建 cfg：
    - 聚类参数：sorted_times, sorted_client, sorted_edge
    - 聚合函数：agg_fn
    - 模型：从 template 深拷贝
    - writer：每个 fl 独立目录
    """
    if fl_name == "x":
        print("Start FL pre-training for [x]")
        model_states = []
        for cid in range(args.num_clients):
            local_model = copy.deepcopy(global_model_template)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
            for ep in range(1, 500 + 1):
                model_state, _ = train(
                    model=local_model,
                    train_loader=train_loaders[cid],
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )
            model_states.append({k: v.cpu() for k, v in model_state.items()})
        print("matrix shape:", np.asarray(heterogeneity).shape)
        print("len(model):", len(model_states))
        sorted_times, sorted_client, sorted_edge = X_cluster(
            heterogeneity, model_states, alpha=args.a
        )
        # if args.device == False:
        #     agg_fn = Sync_Sync
        # else:
        #     agg_fn = Sync_Async
        agg_fn = Sync_Async

    elif fl_name == "fedfleet":
        sorted_times, sorted_client, sorted_edge = fedfleet_cluster(heterogeneity)
        # if args.device == False:
        #     agg_fn = Sync_Sync
        # else:
        #     agg_fn = Sync_Async
        agg_fn = Sync_Async

    elif fl_name == "hieravg":
        sorted_times, sorted_client, sorted_edge = hierfavg_cluster(heterogeneity)
        # agg_fn = Sync_Sync
        agg_fn = Sync_Async

    elif fl_name == "safi":
        sorted_times, sorted_client, sorted_edge = safi_cluster(heterogeneity, comp)
        # if args.device == False:
        #     agg_fn = Sync_Sync
        # else:
        #     agg_fn = Sync_Async
        agg_fn = Sync_Async

    else:
        raise ValueError(f"Unknown fl aggregator: {fl_name}")

    writer = SummaryWriter(log_dir=os.path.join(base_logdir, fl_name))
    cfg = {
        "fl": fl_name,
        "fn": agg_fn,
        "args": (sorted_times, sorted_client, sorted_edge),
        "model": copy.deepcopy(global_model_template),
        "writer": writer,
    }
    return cfg


def run_training_for_cfg(
        cfg,
        args,
        train_loaders,
        test_loader,
        device,
        criterion,
):
    """
    对单个 cfg (即单个 fl 聚合器) 跑完整训练并返回汇总结果。
    """
    fl_name = cfg["fl"]
    writer = cfg["writer"]

    print(f"\n=== [{fl_name}] training start ===")

    best_acc = -1.0
    best_epoch = -1
    best_loss = None

    start_ts = time.time()

    for epoch in range(args.epochs):
        local_states, num_samples = [], []

        # Local training on each client
        for cid in range(args.num_clients):
            local_model = copy.deepcopy(cfg["model"])
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
            for _ in range(args.local_epochs):
                sd, seen = train(
                    model=local_model,
                    train_loader=train_loaders[cid],
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )

            sd_aligned = align_to_global(sd, cfg["model"].state_dict())
            local_states.append({k: v.cpu() for k, v in sd_aligned.items()})
            num_samples.append(seen)

        # Aggregate
        new_state = cfg["fn"](*cfg["args"], local_states, num_samples)
        cfg["model"].load_state_dict(new_state, strict=True)

        # Evaluate
        acc, loss = evaluate(model=cfg["model"], criterion=criterion, data_loader=test_loader, device=device)
        # val_loss = metrics["loss"]
        # val_acc = metrics["acc"]
        print(
            f"[{fl_name}] Epoch {epoch + 1}/{args.epochs} | "
            f"val_loss={loss:.4f} val_acc={acc:.2f}"
        )
        log_eval(writer, "Val", loss, acc, step=epoch)
        #
        # # Track best
        # if val_acc > best_acc:
        #     best_acc = float(val_acc)
        #     best_epoch = int(epoch + 1)
        #     best_loss = float(val_loss)

    elapsed = time.time() - start_ts

    print(f"=== [{fl_name}] training done ===")
    writer.flush()
    writer.close()

    return {
        "fl": fl_name,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "elapsed_sec": float(elapsed),
    }


def align_to_global(local_sd: dict, global_sd: dict) -> dict:
    aligned = {}
    for k, gv in global_sd.items():
        lv = local_sd.get(k, None)
        if lv is not None and isinstance(lv, torch.Tensor) and lv.shape == gv.shape and lv.dtype == gv.dtype:
            aligned[k] = lv
        else:
            aligned[k] = gv.clone()
    return aligned
