from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train

import envTest
from task import divehole_env

import shutil


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='exp',
                    help='exp name for model hozon')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--agent-number', type=int, default=2, help='agent number')
parser.add_argument('--field-size', type=int, default=10, help='field size')
parser.add_argument('--test-span', type=int, default=60, help='test span')
parser.add_argument('--delete-mode', type=bool, default=False, help='delete agent?')
parser.add_argument('--with-premade', type=bool, default=False, help="with premade agent?")
parser.add_argument('--reward-amount1', type=bool, default=False, help="reward dousuru?")
parser.add_argument('--finish-pattern', type=str, default="soft", help="finish no pattern")
parser.add_argument('--random-move', type=bool, default=True, help="Target ga randome move")

if __name__ == '__main__':
    print("start program")
    #os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    mp.set_start_method("spawn")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 環境を宣言
    env = divehole_env.WolfPackAlpha(args)

    # shared_modelをagent数分用意
    shared_model_ary = []
    for i in range(args.agent_number):
        shared_model_ary.append(ActorCritic(args.field_size, env.action_space))
    for i in range(len(shared_model_ary)):
        shared_model_ary[i].share_memory()
    
    # --env-nameと同じ名前のモデルが存在したら読み込む
    if not os.path.isdir("model"):
        os.mkdirs("model/"+args.env_name)
    if os.path.isdir("model/"+args.env_name):
        for i in range(args.agent_number):
            param = torch.load('model/'+args.env_name+'/'+args.env_name+str(i)+'.pth')
            shared_model_ary[i].load_state_dict(param)
    else:
        os.makedirs("model/"+args.env_name)

    # optimizerもagent数分用意
    optimizer_ary = []
    for i in range(len(shared_model_ary)):
        if args.no_shared:
            optimizer_ary.append(None)
        else:
            optimizer_ary.append(my_optim.SharedAdam(shared_model_ary[i].parameters(), lr=args.lr))
            optimizer_ary[i].share_memory()

    # # マルチプロセスの準備
    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # プロセスを宣言
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model_ary, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model_ary, counter, lock, optimizer_ary[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    # testt = 0
    # counterr = 0
    # lockk = 0
    # train(testt,args,shared_model_ary,counterr,lockk,optimizer_ary[i])