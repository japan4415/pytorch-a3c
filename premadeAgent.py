import numpy
import math

class PremadeAgent():

    def __init__(self,myNumber):
        self.reset()
        self.myNumber = myNumber

    def reset(self):
        self.finishFlag = False
        self.targetGoal = 0

    def getAction(self,args,statusA):
        differA = []
        differA2 = []
        # for i in range(args.agent_number):
        #     differA.append(math.sqrt(math.exp((statusA[0][0] - statusA[args.agent_number+i][0]),2) + math.exp((statusA[0][1] - statusA[args.agent_number+i][1]),2)))
        #     differA2.append(math.sqrt(math.exp((statusA[1][0] - statusA[args.agent_number+i][0]),2) + math.exp((statusA[1][1] - statusA[args.agent_number+i][1]),2)))
        self.otherTarget = 0
        # ↑
        if statusA[0][2] == 255 and statusA[0][3] == 255:
            if statusA[0][1] < statusA[2][1]:
                diff0 = statusA[0][1] + args.field_size - statusA[2][1]
            elif statusA[0][1] > statusA[2][1]:
                diff0 = 0
            elif statusA[0][1] > statusA[2][1]:
                diff0 = statusA[0][1] - status[2][1]
            if statusA[0][1] < statusA[3][1]:
                diff1 = statusA[0][1] + args.field_size - statusA[3][1]
            elif statusA[0][1] > statusA[3][1]:
                diff1 = 0
            elif statusA[0][1] > statusA[3][1]:
                diff1 = statusA[0][1] - status[3][1]
            if diff0 >= diff1:
                self.targetGoal = 1
            else:
                self.targetGoal = 1
        # →
        elif statusA[0][4] == 255:
            if statusA[0][0] > statusA[2][0]:
                diff0 = args.field_size - statusA[0][1] + statusA[2][0]
            elif statusA[0][0] > statusA[2][0]:
                diff0 = 0
            elif statusA[0][0] < statusA[2][0]:
                diff0 = statusA[2][0] - statusA[0][0]
            if statusA[0][0] > statusA[3][0]:
                diff0 = args.field_size - statusA[0][1] + statusA[3][0]
            elif statusA[0][0] > statusA[3][0]:
                diff1 = 0
            elif statusA[0][0] < statusA[3][0]:
                diff1 = status[3][0] - statusA[0][0]
            if diff0 >= diff1:
                self.targetGoal = 1
            else:
                self.targetGoal = 1
        # ↓
        elif statusA[0][2] == 255:
            if statusA[0][1] > statusA[2][1]:
                diff0 = args.field_size - statusA[0][1] + statusA[2][1]
            elif statusA[0][1] > statusA[2][1]:
                diff0 = 0
            elif statusA[0][1] < statusA[2][1]:
                diff0 = statusA[2][1] - statusA[0][1]
            if statusA[0][1] > statusA[3][1]:
                diff1 = args.field_size - statusA[0][1] + statusA[3][1]
            elif statusA[0][1] > statusA[3][1]:
                diff1 = 0
            elif statusA[0][1] < statusA[3][1]:
                diff1 = statusA[3][1] - statusA[0][1]
            if diff0 >= diff1:
                self.targetGoal = 1
            else:
                self.targetGoal = 1
        # ←
        elif statusA[0][3] == 255:
            if statusA[0][0] < statusA[2][0]:
                diff0 = args.field_size + statusA[0][1] - statusA[2][0]
            elif statusA[0][0] == statusA[2][0]:
                diff0 = 0
            elif statusA[0][0] > statusA[2][0]:
                diff0 = statusA[0][0] - statusA[2][0]
            if statusA[0][0] < statusA[3][0]:
                diff1 = args.field_size + statusA[0][1] - statusA[3][0]
            elif statusA[0][0] == statusA[3][0]:
                diff1 = 0
            elif statusA[0][0] > statusA[3][0]:
                diff1 = statusA[0][0] - statusA[3][0]
            if diff0 >= diff1:
                self.targetGoal = 1
            else:
                self.targetGoal = 1

        if self.targetGoal == 0:
            xdiff = math.fabs(statusA[1][0] - statusA[2][0])
            ydiff = math.fabs(statusA[1][1] - statusA[2][1])
            if xdiff >= ydiff:
                if statusA[1][0] - statusA[2][0] >=0:
                    move = 2
                else:
                    move = 9
            else:
                if statusA[1][1] - statusA[2][1] >=0:
                    move = 11
                else:
                    move = 18
        elif self.targetGoal == 1:
            xdiff = math.fabs(statusA[1][0] - statusA[3][0])
            ydiff = math.fabs(statusA[1][1] - statusA[3][1])
            if xdiff >= ydiff:
                if statusA[1][0] - statusA[3][0] >=0:
                    move = 2
                else:
                    move = 9
            else:
                if statusA[1][1] - statusA[3][1] >=0:
                    move = 11
                else:
                    move = 18

        return numpy.int(move)
