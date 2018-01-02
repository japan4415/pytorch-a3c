import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

# オリジナルタスクのimport
import envTest


def test(rank, args, shared_model_ary, counter):
    torch.manual_seed(args.seed + rank)

    #env = create_atari_env(args.env_name)
    #env.seed(args.seed + rank)

    env = envTest.create_divehole(2)

    model_ary = []
    for i in range(len(shared_model_ary)):
        #model_ary.append(ActorCritic(env.observation_space.shape[0], env.action_space))
        model_ary.append(ActorCritic(env.field.shape[0], env.action_space))
        model_ary[i].eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum_ary = [0] * len(shared_model_ary)
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
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
            prob = F.softmax(logit)
            action_ary.append(prob.max(1, keepdim=True)[1].data)

        state, reward_ary, done= env.step(action_ary)
        env.rrender(state,counter.value,episode_length,0)
        try:
            reward_ary[0]
        except:
            reward_ary = [reward_ary]
        done = done or episode_length >= args.max_episode_length
        for i in range(len(shared_model_ary)):
            reward_sum_ary[i] += reward_ary[i]

        # a quick hack to prevent the agent from stucking
        actions.append(action_ary)
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            for i in range(len(shared_model_ary)):
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                time.gmtime(time.time() - start_time)),
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum_ary[i], episode_length))
            reward_sum_ary = [0] * len(shared_model_ary)
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
