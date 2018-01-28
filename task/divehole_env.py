import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from numpy import *

from PIL import Image

import os
import os.path

import math

class WolfPackAlphaNeo(gym.Env):

    def __init__(self,args):
        print("WolfPackAlpha inited")
        self.args = args
        self.action_space = 5
        self.goalColor = (255,255,255)
        self.fieldColor = (0,0,0)
        # http://www.geocities.co.jp/HeartLand/8819/webjpcol.html ここを参考に
        self.agentColorAry = [
            [(204,0,0),(12,0,255),(255,255,0),(0,255,65)],
            [(255,0,178),(67,135,233),(240,179,37),(71,234,126)]
        ]
        self.reset()

    def reset(self):
        self.turn = 0
        self.statusAry = np.zeros((3,self.args.agent_number + 1),dtype=int)
        self.statusAry[self.args.agent_number][2] = 0
        xRandomAry = np.random.choice(range(self.args.field_size),self.args.agent_number+1,replace=False)
        yRandomAry = np.random.choice(range(self.args.field_size),self.args.agent_number+1,replace=False)
        for i in range(self.args.agent_number+1):
            self.statusAry[i][0] = xRandomAry[i]
            self.statusAry[i][1] = yRandomAry[i]
        colorRandomAry = np.random.choice(range(4),self.args.agent_number,replace=False)
        for i in range(self.args.agent_number):
            self.statusAry[i][2] = colorRandomAry[i]
        self.makeField()
        return self.fieldAry

    def makeField(self):
        self.fieldAry = np.zeros((self.args.field_size,self.args.field_size,3),dtype=int)
        for i in range(self.args.agent_number):
            # print(self.fieldAry[self.statusAry[i][0]][self.statusAry[i][1]])
            # print(self.statusAry[i][2])
            # print(self.agentColorAry[i][self.statusAry[i][2]])
            self.fieldAry[self.statusAry[i][1]][self.statusAry[i][0]] = self.agentColorAry[i][self.statusAry[i][2]]
        self.fieldAry[self.statusAry[2][1]][self.statusAry[2][0]] = self.goalColor

    def step0(self):
        action = self.premadeMove()
        self.turn += 1
        i = 0
        color = action // 5
        move = action % 5
        if move == 0:
            pass
        elif move == 1:
            if self.checkPosition(self.statusAry[i][0],self.statusAry[i][1]+1):
                self.statusAry[i][1] += 1
            else:
                action = 0
        elif move == 2:
            if self.checkPosition(self.statusAry[i][0]+1,self.statusAry[i][1]):
                self.statusAry[i][0] += 1
            else:
                action = 0
        elif move == 3:
            if self.checkPosition(self.statusAry[i][0],self.statusAry[i][1]-1):
                self.statusAry[i][1] -= 1
            else:
                action = 0
        elif move == 4:
            if self.checkPosition(self.statusAry[i][0]-1,self.statusAry[i][1]):
                self.statusAry[i][0] -= 1
            else:
                action = 0
        # print(self.statusAry)
        self.makeField()
        return self.fieldAry,action

    def step1(self,action):
        color = action[0][0] // 5
        move = action[0][0] % 5
        i=2
        if move == 0:
            pass
        elif move == 1:
            if self.checkPosition(self.statusAry[i][0]+1,self.statusAry[i][1]):
                self.statusAry[i][0] += 1
            else:
                action = 0
        elif move == 2:
            if self.checkPosition(self.statusAry[i][0],self.statusAry[i][1]+1):
                self.statusAry[i][1] += 1
            else:
                action = 0
        elif move == 3:
            if self.checkPosition(self.statusAry[i][0]-1,self.statusAry[i][1]):
                self.statusAry[i][0] -= 1
            else:
                action = 0
        elif move == 4:
            if self.checkPosition(self.statusAry[i][0],self.statusAry[i][1]-1):
                self.statusAry[i][1] -= 1
            else:
                action = 0
        self.checkFinish()
        move = np.random.choice(range(5))
        if (not self.done) and self.args.random_move:
            move = np.random.choice(range(5))
            if move == 0:
                pass
            elif move == 1:
                if self.checkPosition(self.statusAry[self.args.agent_number][0]+1,self.statusAry[self.args.agent_number][1]):
                    self.statusAry[self.args.agent_number][0] += 1
                else:
                    move = 0
            elif move == 2:
                if self.checkPosition(self.statusAry[self.args.agent_number][0],self.statusAry[self.args.agent_number][1]+1):
                    self.statusAry[self.args.agent_number][1] += 1
                else:
                    move = 0
            elif move == 3:
                if self.checkPosition(self.statusAry[self.args.agent_number][0]-1,self.statusAry[self.args.agent_number][1]):
                    self.statusAry[self.args.agent_number][0] -= 1
                else:
                    move = 0
            elif move == 4:
                if self.checkPosition(self.statusAry[self.args.agent_number][0],self.statusAry[self.args.agent_number][1]-1):
                    self.statusAry[self.args.agent_number][1] -= 1
                else:
                    move = 0
        else:
            move = 0
        actionTarget = move
        # print(self.statusAry)
        self.makeField()
        if self.done:
            if self.args.reward_amount1:
                self.reward = 1
            else:
                self.reward = [(self.args.max_episode_length - self.turn) / self.args.max_episode_length] * self.args.agent_number
        else:
            self.reward = 0
            if self.turn >= self.args.max_episode_length:
                self.done = True
        
        return self.fieldAry,self.reward,self.done,action,actionTarget


    def checkPosition(self,x,y):
        position=[x,y]
        position = np.array(position)
        for i in range(self.args.agent_number+1):
            # print('sA')
            # print(self.statusAry[i][0:2])
            # print('position')
            # print(position)
            if np.allclose(self.statusAry[i][0:2],position):
                return False
            if position[0] < 0 or position[0] > self.args.field_size - 1:
                return False
            if position[1] < 0 or position[1] > self.args.field_size - 1:
                return False
        # print(x)
        # print(y)
        # print('True')
        return True

    def checkFinish(self):
        if self.args.finish_pattern == "soft":
            count = 0
            for i in range(self.args.agent_number):
                if self.statusAry[i][0] == self.statusAry[self.args.agent_number][0]+1 and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]:
                    count += 1
                elif self.statusAry[i][0] == self.statusAry[self.args.agent_number][0] and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]+1:
                    count += 1
                elif self.statusAry[i][0] == self.statusAry[self.args.agent_number][0]-1 and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]:
                    count += 1
                elif self.statusAry[i][0] == self.statusAry[self.args.agent_number][0] and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]-1:
                    count += 1
            if count >= self.args.agent_number:
                self.done = True
            else:
                self.done = False

    def premadeMove(self):
        x = self.statusAry[0][0]
        y = self.statusAry[0][1]
        targetPositionAry = [[x,y-1],[x+1,y],[x,y+1],[x-1,y]]
        targetPositionAryTrue = []
        for i in range(len(targetPositionAry)):
            if self.checkPosition(targetPositionAry[i][0],targetPositionAry[i][1]):
                targetPositionAryTrue.append(targetPositionAry[i])
        distanceAry = []
        for i in range(len(targetPositionAryTrue)):
            # print(self.statusAry[0][0])
            # print(targetPositionAryTrue[i][0])
            # print(targetPositionAryTrue)
            xDistance2 = pow((self.statusAry[0][0] - targetPositionAryTrue[i][0]),2)
            # print(xDistance2)
            yDistance2 = pow((self.statusAry[0][1] - targetPositionAryTrue[i][1]),2)
            # print(yDistance2)
            distance = math.sqrt(xDistance2+yDistance2)
            distanceAry.append(distance)
        # print(distanceAry)
        targetX = targetPositionAryTrue[np.argmin(distanceAry)][0]
        targetY = targetPositionAryTrue[np.argmin(distanceAry)][1]
        if abs(targetX - self.statusAry[0][0]) >= abs(targetX - self.statusAry[0][1]):
            if targetX - self.statusAry[0][0] > 0:
                if self.checkPosition(self.statusAry[0][0]+1,self.statusAry[0][1]):
                    move = 2
                else:
                    if targetY - self.statusAry[0][1] > 0:
                        move = 1
                    else:
                        move = 3
            else:
                if self.checkPosition(self.statusAry[0][0]-1,self.statusAry[0][1]):
                    move = 4
                else:
                    if targetY - self.statusAry[0][1] > 0:
                        move = 1
                    else:
                        move = 3
        else:
            if targetY - self.statusAry[0][1] > 0:
                if self.checkPosition(self.statusAry[0][0],self.statusAry[0][1]+1):
                    move = 1
                else:
                    if targetX - self.statusAry[0][0] > 0:
                        move = 2
                    else:
                        move = 4
            else:
                if self.checkPosition(self.statusAry[0][0],self.statusAry[0][1]-1):
                    move = 3
                else:
                    if targetX - self.statusAry[0][0] > 0:
                        move = 2
                    else:
                        move = 4
        if targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[0]:
            color = 0
        elif targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[1]:
            color = 1
        elif targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[2]:
            color = 2
        elif targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[3]:
            color = 3
        return color * 5 + move

    def rrender(self, state,name,stepN,mode):
        if not os.path.isdir('log/no' + str(name)):
            os.makedirs('log/no' + str(name))
        if mode == 0:
            img = Image.fromarray(np.uint8(state))
            img = img.resize((int(img.width * 5),int(img.height * 5)),Image.BOX)
            img.save('log/no' + str(name) + "/" + str(stepN) + '.png')


class WolfPackAlpha(gym.Env):

    def __init__(self,args):
        print("WolfPackAlpha inited")
        self.args = args
        self.action_space = 5
        self.goalColor = (255,255,255)
        self.fieldColor = (0,0,0)
        # http://www.geocities.co.jp/HeartLand/8819/webjpcol.html ここを参考に
        self.agentColorAry = [
            [(204,0,0),(12,0,255),(255,255,0),(0,255,65)],
            [(255,0,178),(67,135,233),(240,179,37),(71,234,126)]
        ]
        self.reset()

    def reset(self):
        self.turn = 0
        self.statusAry = np.zeros((3,self.args.agent_number + 1),dtype=int)
        self.statusAry[self.args.agent_number][2] = 0
        xRandomAry = np.random.choice(range(self.args.field_size),self.args.agent_number+1,replace=False)
        yRandomAry = np.random.choice(range(self.args.field_size),self.args.agent_number+1,replace=False)
        for i in range(self.args.agent_number+1):
            self.statusAry[i][0] = xRandomAry[i]
            self.statusAry[i][1] = yRandomAry[i]
        colorRandomAry = np.random.choice(range(4),self.args.agent_number,replace=False)
        for i in range(self.args.agent_number):
            self.statusAry[i][2] = colorRandomAry[i]
        self.makeField()
        return self.fieldAry

    def makeField(self):
        self.fieldAry = np.zeros((self.args.field_size,self.args.field_size,3),dtype=int)
        for i in range(self.args.agent_number):
            # print(self.fieldAry[self.statusAry[i][0]][self.statusAry[i][1]])
            # print(self.statusAry[i][2])
            # print(self.agentColorAry[i][self.statusAry[i][2]])
            self.fieldAry[self.statusAry[i][0]][self.statusAry[i][1]] = self.agentColorAry[i][self.statusAry[i][2]]
        self.fieldAry[self.statusAry[2][1]][self.statusAry[2][0]] = self.goalColor

    def step(self,actionAry):
        if self.args.with_premade:
            actionAry[0] = self.premadeMove()
        self.turn += 1
        for i in range(self.args.agent_number):
            color = actionAry[i] // 5
            move = actionAry[i] % 5
            if move == 0:
                pass
            elif move == 1:
                if self.checkPosition(self.statusAry[i][0]+1,self.statusAry[i][1]):
                    self.statusAry[i][0] += 1
                else:
                    actionAry[i] = 0
            elif move == 2:
                if self.checkPosition(self.statusAry[i][0],self.statusAry[i][1]+1):
                    self.statusAry[i][1] += 1
                else:
                    actionAry[i] = 0
            elif move == 3:
                if self.checkPosition(self.statusAry[i][0]-1,self.statusAry[i][1]):
                    self.statusAry[i][0] -= 1
                else:
                    actionAry[i] = 0
            elif move == 4:
                if self.checkPosition(self.statusAry[i][0],self.statusAry[i][1]-1):
                    self.statusAry[i][1] -= 1
                else:
                    actionAry[i] = 0
        self.checkFinish()
        move = np.random.choice(range(5))
        if (not self.done) and self.args.random_move:
            move = np.random.choice(range(5))
            if move == 0:
                pass
            elif move == 1:
                if self.checkPosition(self.statusAry[self.args.agent_number][0]+1,self.statusAry[self.args.agent_number][1]):
                    self.statusAry[self.args.agent_number][0] += 1
                else:
                    move = 0
            elif move == 2:
                if self.checkPosition(self.statusAry[self.args.agent_number][0],self.statusAry[self.args.agent_number][1]+1):
                    self.statusAry[self.args.agent_number][1] += 1
                else:
                    move = 0
            elif move == 3:
                if self.checkPosition(self.statusAry[self.args.agent_number][0]-1,self.statusAry[self.args.agent_number][1]):
                    self.statusAry[self.args.agent_number][0] -= 1
                else:
                    move = 0
            elif move == 4:
                if self.checkPosition(self.statusAry[self.args.agent_number][0],self.statusAry[self.args.agent_number][1]-1):
                    self.statusAry[self.args.agent_number][1] -= 1
                else:
                    move = 0
        actionAry.append(move)
        # print(self.statusAry)
        self.makeField()
        if self.done:
            if self.args.reward_amount1:
                self.reward = [1] * self.args.agent_number
            else:
                self.reward = [(self.args.max_episode_length - self.turn) / self.args.max_episode_length] * self.args.agent_number
        else:
            self.reward = [0] * self.args.agent_number
            if self.turn >= self.args.max_episode_length:
                self.done = True
        
        return self.fieldAry,self.reward,self.done,actionAry

    def checkPosition(self,x,y):
        position=[x,y]
        position = np.array(position)
        for i in range(self.args.agent_number+1):
            # print('sA')
            # print(self.statusAry[i][0:2])
            # print('position')
            # print(position)
            if np.allclose(self.statusAry[i][0:2],position):
                return False
            if position[0] < 0 or position[0] > self.args.field_size - 1:
                return False
            if position[1] < 0 or position[1] > self.args.field_size - 1:
                return False
        # print(x)
        # print(y)
        # print('True')
        return True

    def checkFinish(self):
        if self.args.finish_pattern == "soft":
            count = 0
            for i in range(self.args.agent_number):
                if self.statusAry[i][0] == self.statusAry[self.args.agent_number][0]+1 and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]:
                    count += 1
                elif self.statusAry[i][0] == self.statusAry[self.args.agent_number][0] and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]+1:
                    count += 1
                elif self.statusAry[i][0] == self.statusAry[self.args.agent_number][0]-1 and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]:
                    count += 1
                elif self.statusAry[i][0] == self.statusAry[self.args.agent_number][0] and self.statusAry[i][1] == self.statusAry[self.args.agent_number][1]-1:
                    count += 1
            if count >= self.args.agent_number:
                self.done = True
            else:
                self.done = False

    def premadeMove(self):
        x = self.statusAry[self.args.agent_number][0]
        y = self.statusAry[self.args.agent_number][1]
        targetPositionAry = [[x,y-1],[x+1,y],[x,y+1],[x-1,y]]
        targetPositionAryTrue = []
        for i in range(len(targetPositionAry)):
            if self.checkPosition(targetPositionAry[i][0],targetPositionAry[i][1]):
                targetPositionAryTrue.append(targetPositionAry[i])
        distanceAry = []
        for i in range(len(targetPositionAryTrue)):
            # print(self.statusAry[self.args.agent_number][0])
            # print(targetPositionAryTrue[i][0])
            # print(targetPositionAryTrue)
            xDistance2 = pow((self.statusAry[0][0] - targetPositionAryTrue[i][0]),2)
            # print(xDistance2)
            yDistance2 = pow((self.statusAry[0][1] - targetPositionAryTrue[i][1]),2)
            # print(yDistance2)
            distance = math.sqrt(xDistance2+yDistance2)
            distanceAry.append(distance)
        # print(distanceAry)
        targetX = targetPositionAryTrue[np.argmin(distanceAry)][0]
        targetY = targetPositionAryTrue[np.argmin(distanceAry)][1]
        if abs(targetX - self.statusAry[self.args.agent_number][0]) >= abs(targetX - self.statusAry[self.args.agent_number][1]):
            if targetX - self.statusAry[self.args.agent_number][0] > 0:
                if self.checkPosition(self.statusAry[self.args.agent_number][0]+1,self.statusAry[self.args.agent_number][1]):
                    move = 2
                else:
                    if targetY - self.statusAry[self.args.agent_number][1] > 0:
                        move = 1
                    else:
                        move = 3
            else:
                if self.checkPosition(self.statusAry[self.args.agent_number][0]-1,self.statusAry[self.args.agent_number][1]):
                    move = 4
                else:
                    if targetY - self.statusAry[self.args.agent_number][1] > 0:
                        move = 1
                    else:
                        move = 3
        else:
            if targetY - self.statusAry[self.args.agent_number][1] > 0:
                if self.checkPosition(self.statusAry[self.args.agent_number][0],self.statusAry[self.args.agent_number][1]+1):
                    move = 1
                else:
                    if targetX - self.statusAry[self.args.agent_number][0] > 0:
                        move = 2
                    else:
                        move = 4
            else:
                if self.checkPosition(self.statusAry[self.args.agent_number][0],self.statusAry[self.args.agent_number][1]-1):
                    move = 3
                else:
                    if targetX - self.statusAry[self.args.agent_number][0] > 0:
                        move = 2
                    else:
                        move = 4
        if targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[0]:
            color = 0
        elif targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[1]:
            color = 1
        elif targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[2]:
            color = 2
        elif targetPositionAryTrue[np.argmin(distanceAry)] == targetPositionAry[3]:
            color = 3
        return color * 5 + move



class DiveholeEnv(gym.Env):
    metadata = {'render.model':['normal']}

    def __init__(self,args,agentN,fieldSize,turnMax):
        print("env inited")
        self.args = args
        self.availableA = range(20)
        self.action_space = 20
        self.agentN = agentN
        self.fieldSize = fieldSize
        # 色の宣言 黄色 青 赤 黄緑
        self.colorAA = []
        for i in range(self.agentN):
            self.colorAA.append([[255-i*130,255-i*130,0],[0,0,255-i*130],[255-i*130,0,0],[0,255-i*130,0]])
        self.turnMax = turnMax
        self.turnMaxx = turnMax
        self._reset()
    
    def _step(self,AA):
        for i,A in enumerate(AA):
            if self.statusA[i][0] != 102:
                # Aをnumpy形式に変換
                # A = A.numpy()
                # 計算ようにstatusをバックアップ
                statusBA = self.statusA.copy()
                # 動きの種類を計算
                color = A // 5
                move = A % 5
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
                    # print(type(self.colorAA[i][color[0][0]]))
                    self.statusA[i][2:5] = self.colorAA[i][color]
        
        # Fの判定
        fA = [0] * self.agentN
        # ゴールだけ消える版
        # for i in range(self.agentN):
        #     for i2 in range(self.agentN):
        #         #print(i+self.agentN+i2)
        #         if self.statusA[self.agentN+i2][0] == 102:
        #             fA[i2] = 1
        #         else:
        #             if self.statusA[i][0] == self.statusA[self.agentN+i2][0] and self.statusA[i][1] == self.statusA[self.agentN+i2][1]:
        #                 fA[i2] = 1
        #                 if self.args.delete_mode:
        #                     self.statusA[i][0] = 102
        #                     self.statusA[i][1] = 102
        #                 self.statusA[self.agentN+i2][0] = 102
        #                 self.statusA[self.agentN+i2][1] = 102
        # if sum(fA) == self.agentN:
        #     self.F = True
        # else:
        #     self.F = False
        # 両方重なったら終了版
        for i in range(self.agentN):
            for i2 in range(self.agentN):
                if self.statusA[i][0] == self.statusA[self.agentN+i2][0] and self.statusA[i][1] == self.statusA[self.agentN+i2][1]:
                    fA[0] += 1
        if fA[0] == self.agentN and (self.statusA[0][0] != self.statusA[1][0] or self.statusA[0][1] != self.statusA[1][1]):
            #print(fA[0])
            self.F = True
        else:
            self.F = False

        # R計算
        if self.turn < self.turnMax:
            if self.F:
                # R = [(self.turnMax - self.turn) / self.turnMax,(self.turnMax - self.turn) / self.turnMax]
                # # print("yay: "+str(R[0]))
                R = [1,1]
            else:
                R = [0,0]
        else:
            R = [0,0]
            self.F = True

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
            img = img.resize((int(img.width * 5),int(img.height * 5)),Image.BOX)
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