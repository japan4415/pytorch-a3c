import gym
from task import divehole_env

if __name__ == "__main__":
    env = divehole_env.DiveholeEnv(2)
    print(env.statusA)
    env._step([1,2])
    print(env.statusA)

def create_divehole(agentN,fieldSize):
    env = divehole_env.DiveholeEnv(agentN,fieldSize)
    return env