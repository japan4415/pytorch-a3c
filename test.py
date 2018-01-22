import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

# オリジナルタスクのimport
import envTest
from task import divehole_env
import premadeAgent

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from progressbar import ProgressBar
import os
import shutil

def test(rank, args, shared_model_ary, counter):
    print('lets start!!!')
    torch.manual_seed(args.seed + rank)

    # log書き込み環境の整理
    if os.path.isdir("./log"):
        shutil.rmtree("./log")
    os.makedirs('log/'+args.env_name)

    # premade agentが必要な場合宣言
    if args.with_premade:
        pA = premadeAgent.PremadeAgent(1)

    #env = create_atari_env(args.env_name)
    #env.seed(args.seed + rank)

    #env = envTest.create_divehole(args,args.agent_number,args.field_size,args.max_episode_length)
    env = divehole_env.WolfPackAlpha(args)

    model_ary = []
    for i in range(len(shared_model_ary)):
        #model_ary.append(ActorCritic(env.observation_space.shape[0], env.action_space))
        model_ary.append(ActorCritic(args.field_size, env.action_space))
        model_ary[i].eval()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = [deque(maxlen=100)] * len(shared_model_ary)
    # episode_length = 0

    # kai = 1
    test_kai = 0

    # reward合計
    total_reward = 0
    total_total_reward = 0

    # graph表示用
    x = []
    xt = []
    y = []
    yb = []

    test_kaisu = 100
    episode_amount = 0

    # progressbar表示用
    p = ProgressBar(0, test_kaisu)

    while True:

        test_kai += 1
        currentCounter = counter.value
        rewardStore = []

        

        for i in range(len(shared_model_ary)):
            model_ary[i].load_state_dict(shared_model_ary[i].state_dict())

        for i in range(len(shared_model_ary)):
            torch.save(shared_model_ary[i].state_dict(), 'model/'+args.env_name+'/'+'model'+str(i)+'.pth')

        p = ProgressBar(0, test_kaisu)

        for kai in range(test_kaisu):

            episode_amount += 1

            done = False
            hx_ary = []
            cx_ary = []
            for i in range(len(shared_model_ary)):
                cx_ary.append(Variable(torch.zeros(1, 256), volatile=True))
                hx_ary.append(Variable(torch.zeros(1, 256), volatile=True))
            episode_length = 0
            state = env.reset()
            for i in range(len(shared_model_ary)):
                actions[i].clear()

            while done == False:
                episode_length += 1
                action_ary = []

                # 行動選択
                state = torch.from_numpy(state)
                # print(state)
                for i in range(len(shared_model_ary)):
                    value, logit, (hx_ary[i], cx_ary[i]) = model_ary[i]((Variable(state.unsqueeze(0), volatile=True), (hx_ary[i], cx_ary[i])))
                    prob = F.softmax(logit,dim=1)
                    action_ary.append(prob.max(1, keepdim=True)[1].data.numpy())
                if args.with_premade:
                    action_ary[1] = pA.getAction(args,env.statusA)

                state, reward_ary, done= env.step(action_ary)
                done = done or episode_length >= env.turnMaxx

                for i in range(len(shared_model_ary)):
                    # スタックしていた場合，終了
                    actions[i].append(action_ary[i])
                    if actions[i].count(actions[i][0]) == actions[i].maxlen:
                        done = True
                        episode_length = env.turnMaxx
                    # logファイルにstate書き込み
                    f=open('log/'+args.env_name+'/state/'+currentCounter+'/'+episode_amount+'.log','w')
                    text = str(env.turn-1) + ' '
                    for j in range(len(shared_model_ary)+1):
                        text = text + str(env.stateAry[j][0]) + ' ' + str(env.stateAry[j][1]) +  ' ' + str(env.stateAry[j][2]) +  ' '
                    text = text + '\n'
                    f.writelines(text)
                    f.close()
                    # logファイルにmove書き込み
                    f=open('log/'+args.env_name+'/move/'+currentCounter+'/'+episode_amount+'.log','w')
                    text = str(env.turn-1) + ' ' + str(action_ary[0]) + ' ' + str(action_ary[1]) +  ' ' + str(action_ary[2]) +'\n'
                    f.writelines(text)
                    f.close()

                if kai==0:
                    env.rrender(state,kai,episode_length,0)

                if episode_length != env.turnMaxx and reward_ary[0] > 0:
                    pass
                    #print("maybe gall, kai: {}, length: {}, reward: {}".format(kai,episode_length,reward_ary[0]))

            rewardStore.append(episode_length)
            # logファイルにresult書き込み
            f=open('log/'+args.env_name+'/result.log','w')
            text = str(episode_amount) + ' ' + str(currentCounter) + ' ' + str(episode_length) +'\n'
            f.writelines(text)
            f.close() 
            p.update(kai)

        print("Time {}, Kai {}, num steps {}, FPS {:.0f}, reward/10 {}".format(
                    time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
                    test_kai,
                    currentCounter,
                    currentCounter / (time.time() - start_time),
                    sum(rewardStore) / test_kaisu
                    ))
        x.append(currentCounter)
        y.append(sum(rewardStore) / test_kaisu)
        bunsan = 0
        for j in rewardStore:
            bunsan += pow((j - sum(rewardStore) / test_kaisu),2)
        bunsan = bunsan / test_kaisu
        yb.append(bunsan)
        plt.clf()
        plt.plot(x,y,marker=".")
        plt.savefig("log/graph.png")
        plt.clf()
        plt.plot(x,yb,marker=".")
        plt.savefig("log/graph2.png")

        # # print("Test Start")
        # episode_length += 1
        # # Sync with the shared model
        # if done:
        #     hx_ary = []
        #     cx_ary = []
        #     for i in range(len(shared_model_ary)):
        #         model_ary[i].load_state_dict(shared_model_ary[i].state_dict())
        #         cx_ary.append(Variable(torch.zeros(1, 256), volatile=True))
        #         hx_ary.append(Variable(torch.zeros(1, 256), volatile=True))
        # else:
        #     for i in range(len(shared_model_ary)):
        #         cx_ary[i] = Variable(cx_ary[i].data, volatile=True)
        #         hx_ary[i] = Variable(hx_ary[i].data, volatile=True)

        # # action selectの部分
        # action_ary = []
        # for i in range(len(shared_model_ary)):
        #     #print(state)
        #     value, logit, (hx_ary[i], cx_ary[i]) = model_ary[i]((Variable(state.unsqueeze(0), volatile=True), (hx_ary[i], cx_ary[i])))
        #     prob = F.softmax(logit,dim=1)
        #     action_ary.append(prob.max(1, keepdim=True)[1].data.numpy())

        # if args.with_premade:
        #     action_ary[1] = pA.getAction(args,env.statusA)

        # state, reward_ary, done= env.step(action_ary)
    

        # # try:
        # #     reward_ary[0]
        # # except:
        # #     reward_ary = [reward_ary]
        # # print(episode_length)
        # # print(done)


        # done = done or episode_length >= env.turnMaxx

        # # a quick hack to prevent the agent from stucking
        # for i in range(len(shared_model_ary)):
        #     actions[i].append(action_ary[i])
        #     if actions[i].count(actions[i][0]) == actions[i].maxlen:
        #         done = True
        #         # print('repeat max')
        # if kai%10==0:
        #     env.rrender(state,kai,episode_length,0)

        # if done:
        #     total_reward += reward_ary[0]
        #     total_total_reward += reward_ary[0]
        #     # graph出力
        #     x.append(counter.value)
        #     xt.append(time.time() - start_time)
        #     y.append(reward_ary[0])
        #     yt.append(total_total_reward/kai)
        #     plt.clf()
        #     plt.plot(x,y,marker=".")
        #     plt.savefig("log/graph.png")
        #     plt.clf()
        #     plt.plot(xt,y,marker=".")
        #     plt.savefig("log/graph2.png")
        #     plt.clf()
        #     plt.plot(x,yt,marker=".")
        #     plt.savefig("log/graph3.png")

        #     if episode_length != env.turnMaxx and reward_ary[0] > 0:
        #         print("maybe gall, kai: {}, length: {}, reward: {}".format(kai,episode_length,reward_ary[0]))
        #     if kai%10 == 0:
        #         print("Time {}, Kai {}, num steps {}, FPS {:.0f}, reward/10 {}, episode length {}".format(
        #             time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
        #             kai,
        #             counter.value, 
        #             counter.value / (time.time() - start_time),
        #             total_reward / 10, 
        #             episode_length))
        #         total_reward = 0
        #         p = ProgressBar(1, 10)
        #         for i in range(len(shared_model_ary)):
        #             torch.save(shared_model_ary[i].state_dict(), 'model/'+args.env_name+str(i)+".pth")
        #         time.sleep(args.test_span)
        #     episode_length = 0
        #     for i in range(len(shared_model_ary)):
        #         actions[i].clear()
        #     state = env.reset()
        #     # time.sleep(args.test_span)
        #     # print("sleep")
        #     kai += 1
        #     p.update(kai%10+1)

        # state = torch.from_numpy(state)
