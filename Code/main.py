import copy

import torch

from federated import run_training_for_cfg, build_cfg_for_fl, get_base_logdir
from heterogeneity import heterogeneous_matrix
from model import LateFusionConcat
from option import args_parser
from utddata import build_federated_clients_pkl
from utddata import UTDDataset as UTDdataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is ", device)

    print("The global epoch is ", args.epochs)
    print("The learning rate is ", args.lr)

    client_loaders, test_loader, owned, client_indices = build_federated_clients_pkl(
        args=args,
        train_pkl='data/trainUTD.pkl',
        test_pkl='data/testUTD.pkl',
        num_clients=args.num_clients,
        alpha=args.alpha_de,
        missing_prob=args.ownership_prob,
        batch_size=args.batch_size,
        seed=args.seed,
        min_size=args.min_size,
        ensure_at_least_one=True,
    )

    # global_model = BlockTrainerBlend(args=args, dev=device).to(device)

    global_model = LateFusionConcat(
        mm_dim=1600,
        hidden_dim=2048,
        dropout=0.3,
        num_classes=27 
    ).to(device)
    
    init_state = copy.deepcopy(global_model.state_dict())
    matrix_heterogeneity, matrix_comp = heterogeneous_matrix(args)
    base_logdir = get_base_logdir(args)
    criterion = torch.nn.CrossEntropyLoss()

    # fl_list = ["x","safi","hieravg","fedfleet"]
    # fl_list = ["x", "hieravg"]
    # fl_list = ["safi"]
    # fl_list = ["hieravg"]
    fl_list = ["x"]
    # fl_list = ["fedfleet"]
    # fl_list = args.fl
    results = []
    for fl_name in fl_list:
        global_model.load_state_dict(init_state, strict=True)
        global_model_template = copy.deepcopy(global_model)
        cfg = build_cfg_for_fl(
            fl_name=fl_name,
            args=args,
            global_model_template=global_model,
            train_loaders=client_loaders,
            heterogeneity=matrix_heterogeneity,
            comp=matrix_comp,
            device=device,
            criterion=criterion,
            base_logdir=base_logdir,
        )

        r = run_training_for_cfg(
            cfg=cfg,
            args=args,
            train_loaders=client_loaders,
            test_loader=test_loader,
            device=device,
            criterion=criterion,
        )

        results.append(r)

