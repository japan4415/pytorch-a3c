import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic
from task import divehole_env

import envTest

import premadeAgent


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model_ary, counter, lock, optimizer=None):
    # torchのseedを定める
    torch.manual_seed(args.seed + rank)

    # 環境を宣言
    #env = create_atari_env(args.env_name)
    env = divehole_env.WolfPackAlphaNeo(args)
    #env.seed(args.seed + rank)

    # モデルの宣言
    model_ary = []
    for i in range(len(shared_model_ary)):
        #model_ary.append(ActorCritic(env.observation_space.shape[0], env.action_space))
        model_ary.append(ActorCritic(env.args.field_size, env.action_space))

    # オプティマイザーの宣言
    optimizer_ary = [None] * len(shared_model_ary)
    for i in range(len(shared_model_ary)):
        if optimizer_ary[i] is None:
            optimizer_ary[i] = optim.Adam(shared_model_ary[i].parameters(), lr=args.lr)

    # ???
    for i in range(len(shared_model_ary)):
        model_ary[i].train()

    # premade agentが必要な場合宣言
    # if args.with_premade:
    #     pA = premadeAgent.PremadeAgent(1)

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        # shared_modelのロード
        for i in range(len(shared_model_ary)):
            model_ary[i].load_state_dict(shared_model_ary[i].state_dict())
        if done:
            cx_ary = []
            hx_ary = []
            for i in range(len(shared_model_ary)):
                cx_ary.append(Variable(torch.zeros(1, 256)))
                hx_ary.append(Variable(torch.zeros(1, 256)))
        else:
            for i in range(len(shared_model_ary)):
                cx_ary[i] = Variable(cx.data)
                hx_ary[i] = Variable(hx.data)

        # 格納場所の宣言
        values_ary = []
        log_probs_ary = []
        rewards_ary = []
        entropies_ary = []
        for i in range(len(shared_model_ary)):
            values_ary.append([])
            log_probs_ary.append([])
            rewards_ary.append([])
            entropies_ary.append([])

        # 経験蓄積開始
        for step in range(args.num_steps):
            episode_length += 1

            # 行動を計算？
            action_ary = []
            value_ary = []
            log_prob_ary = []

            i=1
            value, logit, (hx, cx) = model_ary[i]((Variable(state.unsqueeze(0)),(hx_ary[i], cx_ary[i])))
            prob = F.softmax(logit,dim=1)
            log_prob = F.log_softmax(logit,dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies_ary[i].append(entropy)

            action = prob.multinomial().data.numpy()
            log_prob_ary.append(log_prob.gather(1, Variable(prob.multinomial().data)))


            state, actionPremade = env.step0()
            # if args.with_premade:
            #     action_ary[1] = pA.getAction(args,env.statusAry)

            # 実行してs,r,dを受け取る
            state, reward_ary, done, act_action_ary = env.step1(action)
            # if reward_ary[0] > 0:
            #     print(reward_ary)
            done = done or episode_length >= args.max_episode_length


            #print(reward_ary)
            for i in range(len(shared_model_ary)):
                reward_ary[i] = max(min(reward_ary[i], 1), -1)

            # ここまでで問題発生
            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            for i in range(len(shared_model_ary)):
                values_ary[i].append(value)
                log_probs_ary[i].append(log_prob_ary[i])
                rewards_ary[i].append(reward_ary[i])

            if done:
                break

        R_ary = [torch.zeros(1, 1)] * len(shared_model_ary)
        i=1
        if not done:
            value, _, _ = model_ary[i]((Variable(state.unsqueeze(0)), (hx_ary[i], cx_ary[i])))
            R_ary[i] = value.data

        values_ary[i].append(Variable(R_ary[i]))

        policy_loss_ary = [0] * len(shared_model_ary)
        value_loss_ary = [0] * len(shared_model_ary)

        R_ary[i] = Variable(R_ary[i])

        gae_ary = [torch.zeros(1, 1)] * len(shared_model_ary)
        j=1
        for i in reversed(range(len(rewards_ary[j]))):
            R_ary[j] = args.gamma * R_ary[j] + rewards_ary[j][i]
            advantage = R_ary[j] - values_ary[j][i]
            value_loss_ary[j] = value_loss_ary[j] + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards_ary[j][i] + args.gamma * values_ary[j][i + 1].data - values_ary[j][i].data
            gae_ary[j] = gae_ary[j] * args.gamma * args.tau + delta_t

            policy_loss_ary[j] = policy_loss_ary[j] - log_probs_ary[j][i] * Variable(gae_ary[j]) - args.entropy_coef * entropies_ary[j][i]

        optimizer_ary[j].zero_grad()

        (policy_loss_ary[j] + args.value_loss_coef * value_loss_ary[j]).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(model_ary[j].parameters(), args.max_grad_norm)

        ensure_shared_grads(model_ary[j], shared_model_ary[j])
        optimizer_ary[j].step()
