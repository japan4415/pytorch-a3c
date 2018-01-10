import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from numpy import *

from PIL import Image

import os
import os.path

class DiveholeEnv(gym.Env):
    metadata = {'render.model':['normal']}

    def __init__(self,agentN,fieldSize):
        print("env inited")
        self.availableA = range(20)
        self.action_space = 20
        self.agentN = agentN
        self.fieldSize = fieldSize
        # 色の宣言 黄色 青 赤 黄緑
        self.colorAA = []
        for i in range(self.agentN):
            self.colorAA.append([[255-i,255-i,0],[0,0,255-i],[255-i,0,0],[0,255-i,0]])
        self.turnMax = 10000
        self.turnMaxx = 10000
        self._reset()
    
    def _step(self,AA):
        for i,A in enumerate(AA):
            if self.statusA[i][0] != 102:
                # Aをnumpy形式に変換
                A = A.numpy()
                # 計算ようにstatusをバックアップ
                statusBA = self.statusA.copy()
                # 動きの種類を計算
                color = A // 5
                move = A % 5
                print(move)
                # 移動をstatusAに反映
                if move == 0:
                    pass
                elif move == 1:
                    self.statusA[i][0] += 1
                elif move == 2:
                    self.statusA[i][1] += 1
                elif move == 3:
                    self.statusA[i][0] -= 1
                elif move == 4:
                    self.statusA[i][1] -= 1
                # 位置を正しく
                if self.statusA[i][0] == -1:
                    self.statusA[i][0] = self.fieldSize-1
                elif self.statusA[i][0] == self.fieldSize:
                    self.statusA[i][0] = 0
                if self.statusA[i][1] == -1:
                    self.statusA[i][1] = self.fieldSize-1
                elif self.statusA[i][1] == self.fieldSize:
                    self.statusA[i][1] = 0
                # 色をstatusAに反映
                if self.statusA[i][0] != 102:
                    # print(i)
                    # print("color")
                    # print(color)
                    # print(self.colorAA[i][color[0][0]])
                    self.statusA[i][2:5] = self.colorAA[i][color[0][0]]
        
        # Fの判定
        fA = [0] * self.agentN
        for i in range(self.agentN):
            for i2 in range(self.agentN):
                #print(i+self.agentN+i2)
                if self.statusA[self.agentN+i2][0] == 102:
                    fA[i2] = 1
                else:
                    if self.statusA[i][0] == self.statusA[self.agentN+i2][0] and self.statusA[i][1] == self.statusA[self.agentN+i2][1]:
                        fA[i2] = 1
                        self.statusA[self.agentN+i2][0] = 102
                        self.statusA[self.agentN+i2][1] = 102
        if sum(fA) == self.agentN:
            self.F = True
        else:
            self.F = False

        # R計算
        if self.turn <= self.turnMax:
            if self.F:
                R = [(self.turnMax - self.turn) / self.turnMax,(self.turnMax - self.turn) / self.turnMax]
                # print("yay: "+str(R[0]))
            else:
                R = [0,0]
        else:
            R = [0,0]
            self.F = False

        # 再レンダリング
        self.field = np.zeros(((self.fieldSize,self.fieldSize,3)))
        for status in self.statusA:
            if status[0] != 102:
                self.field[status[0]][status[1]] = status[2:5]

        self.turn += 1

        return self.field,R,self.F


    def _reset(self):
        self.turn = 1
        self.F = [0] * self.agentN
        self.statusA = np.zeros((self.agentN*2,2),dtype="int32")
        positionXYA = [random.choice(range(self.fieldSize),self.agentN*2,replace=False),random.choice(range(self.fieldSize),self.agentN*2,replace=False)]
        colorA = random.choice(range(4),self.agentN,replace=False)
        for i in range(self.agentN*2):
            self.statusA[i,0] = positionXYA[0][i]
            self.statusA[i,1] = positionXYA[1][i]
        colorAG = np.empty((0,3),int)
        for i in range(self.agentN):
            colorAG = np.append(colorAG,[self.colorAA[i][colorA[i]]],axis=0)
        for i in range(self.agentN):
            colorAG = np.append(colorAG,[[255,255,255]],axis=0)
        # print(self.statusA.shape)
        # print(colorAG.shape)
        # print(self.statusA)
        # print(colorAG)
        self.statusA = np.c_[self.statusA,colorAG]
        # print(self.statusA)
        self.field = np.zeros(((self.fieldSize,self.fieldSize,3)))
        # print(self.field)
        for status in self.statusA:
            if status[0] != 102:
                self.field[status[0]][status[1]] = status[2:5]
        # print (self.field)

        return self.field


    def rrender(self, state,name,stepN,mode):
        if not os.path.isdir('log/no' + str(name)):
            os.makedirs('log/no' + str(name))
        if mode == 0:
            img = Image.fromarray(np.uint8(state))
            img = img.resize((int(img.width * 5),int(img.height * 5)),Image.LANCZOS)
            img.save('log/no' + str(name) + "/" + str(stepN) + '.png')


if __name__ == "__main__":
    divehole = DiveholeEnv(1)
    img = divehole.reset()
    img = Image.fromarray(np.uint8(img))
    img = img.resize((int(img.width * 5),int(img.height * 5)),Image.LANCZOS)
    img.save('log/0.png')
    for i in range(150):
        state,_,_ = divehole.step([random.choice(range(20))])
        img = Image.fromarray(np.uint8(state))
        img = img.resize((int(img.width * 5),int(img.height * 5)),Image.LANCZOS)
        img.save('log/'+str(i+1)+'.png')