import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha', help='select from [bitcoin_alpha, intrusion, ppi, epinions, reddit, mag]')
    parser.add_argument('--task', type=int, default=0, help='0: binary, 1: multi-class')    
    
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.4)
    parser.add_argument('--label_ratio', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='GCN', help='select from [GCN, GAT, Cheb]')
    parser.add_argument('--n_embed', type=int, default=64)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_out', type=int, default=32)
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=1028)
    parser.add_argument('--runs', type=int, default=3)

    parser.add_argument('--method', type=str, default='tq', 
                        help='choose from [none, qw, tw, tq]')
    
    parser.add_argument('--n', type=int, default=1, help='number of hop')
    parser.add_argument('--T', type=float, default=0.5, help='temperature of reweight')
    parser.add_argument('--f', type=float, default=0.0001, help='Influence of topological reweight in loss')
    
    parser.add_argument('--mixup', type=int, default=2, help='0: none, 1: topo k fix-mixup, 2: topo k random mixup')
    parser.add_argument('--alpha', type=float, default=4.0, help = 'mixup alpha hyperparamter')
    parser.add_argument('--k', type=float, default=0, help='topo k hyperparamter in mixup')
    parser.add_argument('--h', type=float, default=0, help = 'mixup loss factor')
    
    parser.add_argument('--beta', type=float, default=0.95, help = 'multi-hop ratio weight decay factor')
    parser.add_argument('--w', type=float, default=0.75, help='local homophily ratio weight decay factor')

    return parser.parse_args()