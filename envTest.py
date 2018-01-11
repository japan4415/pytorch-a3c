import gym
from task import divehole_env

if __name__ == "__main__":
    env = divehole_env.DiveholeEnv(2)
    print(env.statusA)
    env._step([1,2])
    print(env.statusA)

def create_divehole(args,agentN,fieldSize,maxTurn):
    env = divehole_env.DiveholeEnv(args,agentN,fieldSize,maxTurn)
    return env