import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

# オリジナルタスクのimport
import envTest

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt




def test(rank, args, shared_model_ary, counter):
    print('lets start!!!')
    torch.manual_seed(args.seed + rank)

    #env = create_atari_env(args.env_name)
    #env.seed(args.seed + rank)

    env = envTest.create_divehole(args.agent_number,args.field_size)

    model_ary = []
    for i in range(len(shared_model_ary)):
        #model_ary.append(ActorCritic(env.observation_space.shape[0], env.action_space))
        model_ary.append(ActorCritic(env.field.shape[0], env.action_space))
        model_ary[i].eval()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = [deque(maxlen=100)] * len(shared_model_ary)
    episode_length = 0

    kai = 0

    # reward合計
    total_reward = 0

    # graph表示用
    x = []
    xt = []
    y = []

    while True:
        print("Test Start")
        episode_length += 1
        # Sync with the shared model
        if done:
            hx_ary = []
            cx_ary = []
            for i in range(len(shared_model_ary)):
                model_ary[i].load_state_dict(shared_model_ary[i].state_dict())
                cx_ary.append(Variable(torch.zeros(1, 256), volatile=True))
                hx_ary.append(Variable(torch.zeros(1, 256), volatile=True))
        else:
            for i in range(len(shared_model_ary)):
                cx_ary[i] = Variable(cx_ary[i].data, volatile=True)
                hx_ary[i] = Variable(hx_ary[i].data, volatile=True)

        # action selectの部分
        action_ary = []
        for i in range(len(shared_model_ary)):
            #print(state)
            value, logit, (hx_ary[i], cx_ary[i]) = model_ary[i]((Variable(state.unsqueeze(0), volatile=True), (hx_ary[i], cx_ary[i])))
            prob = F.softmax(logit,dim=1)
            action_ary.append(prob.max(1, keepdim=True)[1].data)

        state, reward_ary, done= env.step(action_ary)
    

        # try:
        #     reward_ary[0]
        # except:
        #     reward_ary = [reward_ary]
        print(episode_length)
        print(done)


        done = done or episode_length >= env.turnMaxx

        # a quick hack to prevent the agent from stucking
        for i in range(len(shared_model_ary)):
            actions[i].append(action_ary[i].numpy())
            if actions[i].count(actions[i][0]) == actions[i].maxlen:
                done = True
                # print('repeat max')
        if kai%10==0:
            env.rrender(state,kai,episode_length,0)

        if done:
            total_reward += reward_ary[0]
            # graph出力
            x.append(counter.value)
            xt.append(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)))
            y.append(reward_ary[0])
            plt.clf()
            plt.plot(x,y,marker=".")
            plt.savefig("log/graph.png")
            plt.clf()
            plt.plot(xt,y,marker=".")
            plt.savefig("log/graph2.png")

            if episode_length != env.turnMaxx and reward_ary[0] > 0:
                print("maybe gall, kai: {}, length: {}, reward: {}".format(kai,episode_length,reward_ary[0]))
            if kai%10 == 0:
                print("Time {}, num steps {}, FPS {:.0f}, reward/10 {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
                    counter.value, 
                    counter.value / (time.time() - start_time),
                    total_reward / 10, 
                    episode_length))
                total_reward = 0
            episode_length = 0
            for i in range(len(shared_model_ary)):
                actions[i].clear()
            state = env.reset()
            time.sleep(args.test_span)
            print("sleep")
            kai += 1

        state = torch.from_numpy(state)
