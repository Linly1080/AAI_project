import argparse
from datetime import datetime

import torch
import torch.nn as nn
from termcolor import colored
from torch.utils.data import DataLoader

from data import get_dataset, EnvSampler
from model import get_model
from tofu import test_loop
from tofu.utils import to_cuda, squeeze_batch


def get_parser():
    parser = argparse.ArgumentParser(description='TOFU: transfer of unstable features')
    parser.add_argument('--cuda', type=int, default=0)

    # data sample
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='max number of epochs to run')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='sample num_batches batches for each epoch')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)

    # model
    parser.add_argument('--hidden_dim', type=int, default=300)

    #dataset
    parser.add_argument('--src_dataset', type=str, default='')
    parser.add_argument('--tar_dataset', type=str, default='')
    parser.add_argument('--test_dataset', type=str, default='')
    parser.add_argument('--dataset', type=str, default='', help='placeholder')

    # method specification
    parser.add_argument('--num_clusters', type=int, default=2)
    parser.add_argument('--transfer_ebd', action='store_true', default=False,
        help='whether to transfer the ebd function learned from the source task')

    #optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--thres', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    args.dataset = args.test_dataset
    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f' Loading target task {args.dataset}',
          flush=True)
    test_data = get_dataset(args.dataset, is_target=True)
    # initialize model and optimizer based on the dataset
    model, opt = get_model(args, test_data)

    # 加载模型参数
    model['ebd'].load_state_dict(torch.load('src/model_ebd.pth'))
    model['clf'].load_state_dict(torch.load('src/model_clf.pth'))

    test_env=0
    test_loader = DataLoader(
        test_data,
        sampler=EnvSampler(-1, args.batch_size, test_env,
                           test_data.envs[test_env]['idx_list']),
        num_workers=10)

    test_res = test_loop(test_loader, model, args,
                         att_idx_dict=test_data.test_att_idx_dict)

    if 'avg_acc' not in test_res:
        print('Test results: '
              f'{colored("acc", "yellow")} {test_res["acc"]:.4f} '
              f'{colored("loss", "blue")} {test_res["loss"]:.4f} ')
    else:
        print('Test results: '
              f'{colored("worst acc", "yellow")} {test_res["acc"]:.4f} '
              f'{colored("avg acc", "yellow")} {test_res["acc"]:.4f} '
              f'{colored("loss", "blue")} {test_res["loss"]:.4f} ')