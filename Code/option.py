import argparse
import json


def args_parser():
    parser = argparse.ArgumentParser()
    # training arguments
    # parser.add_argument('--dataset', type=str, default='flash', help="dataset choose")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size: B")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    parser.add_argument('--min_size', type=int, default=10, help="min size")
    parser.add_argument('--keep', type=float, nargs='+', default=[0, 1, 2, 3, 4],
                        help="label for dataset")
    parser.add_argument('--a', type=float, default=0.5, help="")

    parser.add_argument('--hidDim',type=int,default=100)
    parser.add_argument('--chunks',type=int,default=4)
    parser.add_argument('--dropIn',type=float,default=0.,)
    parser.add_argument('--dropOut',type=float,default=0.)
    parser.add_argument('--rank',type=int,default=4)
    parser.add_argument('--isBlockAtt', default='True')
    parser.add_argument('--isBlend', type=int, default=1, help='1:True;0:False')

    # federated arguments
    parser.add_argument('--local_epochs', type=int, default=5, help="local training")
    parser.add_argument('--fl', type=str, default='hieravg', help="typy of fl algorithms")
    parser.add_argument('--num_clients', type=int, default=50, help="number of users")
    parser.add_argument('--num_edges', type=int, default=5, help="number of edge aggregator")


    # heterogeneity setting
    parser.add_argument('--data', action='store_true', help="data heterogeneity")
    parser.add_argument('--alpha_de', type=float, default=1.0, help="direhelet")
    parser.add_argument('--device', action='store_true', help="device heterogeneity")
    parser.add_argument('--var', type=float, default=0.125, help="direhelet")
    parser.add_argument('--mean', type=float, default=0.5, help="direhelet")
    parser.add_argument('--modality', action='store_true', help="modality heterogeneity")
    parser.add_argument('--ownership_prob', type=json.loads, default='{"depth": 0.5, "skeleton": 0.5, "inertial": 0.5}')
    args = parser.parse_args()
    return args
