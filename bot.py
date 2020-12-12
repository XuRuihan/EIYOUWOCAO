import json
import random
from MahjongGB import MahjongFanCalculator
"""某个运行实例的输入。每行代表一个step
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2","2 F2","3 0 PLAY F1","3 1 DRAW","3 1 PLAY T4"],"responses":["PASS","PASS","PLAY F1","PASS","PASS"]}
"""

# 定义 request 常量
SELF_DRAW = "2"
OTHERS_TURN = "3"


class Mahjong():
    def __init__(self):
        self.reset()

    def reset(self):  # 用于后期训练 RL 模型的接口
        self.turnID = -1
        self.menFeng = 0
        self.quanFeng = 0

        self.pack = []  # ((packType, tileCode, data), ...),
        self.hand = []
        # self.gang = []  # (牌代码, 1 上家供牌 / 2 对家供牌 / 3 下家供牌)
        # self.peng = []  # 同上
        # self.chi = []  # (中间牌代码, 第 1 / 2 / 3 张是上家供牌)

        self.play = {}  # 记录已经打出的牌
        self.left = 88  # 初始牌墙中的牌数

    def calculateFan(self, state):
        try:
            fans = MahjongFanCalculator(
                self.pack,  # 明牌
                self.hand,  # 暗牌
                state["tileCode"],  # 和的那张牌
                0,  # 补花数，在本模型中应当设为 0
                state["SELF_DRAW"],  # 自摸
                state["isJUEZHANG"],  # 绝张
                state["isGANG"],  # 抢杠和/杠上开花
                state["isLAST"],  # 牌墙最后一张，妙手回春/海底捞月
                self.menFeng,  # 门风
                self.quanFeng,  # 圈风
            )
            totalFan = sum(fan[0] for fan in fans)  # fan[0] 为番数，fan[1] 为番形名称
            return totalFan
        except Exception as e:
            return 0  # 表示没有赢，因此番数为 0

    def review(self):
        '''根据输入的 request 复盘
        Args:
            命令行输入
        Return:
            字符串列表，为当前输入的 request 的切分
        '''
        inputJson = json.loads(input())
        requests, responses = inputJson["requests"], inputJson["responses"]
        self.turnID = len(responses)
        if self.turnID == 0:  # 记录 playerID 和 quan （playerID和门风是否相同？）
            _, self.menFeng, self.quanFeng = tuple(
                map(int, requests[0].split()))
        elif self.turnID == 1:  # 起始手牌，只有在跨回合运行的bot中才能用上
            self.hand = requests[1].split()[5:18]
        else:
            self.hand = requests[1].split()[5:18]
            for i in range(2, self.turnID):
                req = requests[i].split()
                res = responses[i].split()
                if req[0] == SELF_DRAW:  # 摸牌
                    self.left -= 1
                    self.hand.append(req[1])
                    self.hand.remove(res[1])
                elif req[0] == OTHERS_TURN:
                    if req[2] == "PLAY":
                        # 记录打出的牌
                        self.play[req[3]] = self.play.get(req[3], 0) + 1
                    elif req[2] == "DRAW":
                        self.left -= 1
        req = requests[-1].split()  # 当前回合输入
        return req

    def getState(self):  # 读取数据并转换成当前状态
        '''获取当前对局状态
        Args:
            命令行输入
        Return:
            字典，包含当前 request 解析后的状态
        '''
        req = self.review()  # 复盘之前的状态
        # 以下获取当前的状态
        state = {
            "SELF_DRAW": False,  # 当前是否摸牌
            "OTHERS_PLAY": False,  # 当前是否有其他人出牌（需要判断碰/杠/吃/和）
            "tileCode": "",  # 牌形
            "isJUEZHANG": False,  # 绝张
            "isGANG": False,  # 杠
            "isLAST": False,  # 最后一张
        }
        if req[0] == SELF_DRAW:
            state["SELF_DRAW"] = True
            state["tileCode"] == req[1]
            state["isZIMO"] == True
            state["isJUEZHANG"] = (self.play.get(req[1]) == 3)
            state["isLAST"] = (self.left == 1)
        elif req[0] == OTHERS_TURN:
            if req[2] == "PLAY" and req[1] != self.menFeng:
                state["tileCode"] == req[3]
                state["OTHERS_PLAY"] == True
        return state

    def getAction(self, state):
        if self.turnID < 2:
            return "PASS"
        if state["SELF_DRAW"]:  # 自摸
            playCard = random.choice(self.hand)
            return "PLAY " + playCard
        elif state["OTHERS_PLAY"]:  # 其他家出牌
            ### TODO
            return "PASS"
        else:  # 否则无需采取行动
            return "PASS"

    # 合法性检查
    def checkValid(self, action):
        # 上回合输入条件 / 本回合操作 / 下回合的输入
        res = action.split()
        if res[0] == "PLAY":
            # PLAY T6
            # 摸牌 / 打出的牌在手牌中 /
            assert res[1] in self.hand
        elif res[0] == "GANG":
            # GANG T6
            # 摸牌 / 已有三张，杠 / 摸牌

            # GANG
            # 其他玩家打牌 / 已有三张，杠 / 摸牌
            pass
        elif res[0] == "BUGANG":
            # BUGANG T6
            # 摸牌 / 已经有碰，补杠 / 摸牌
            pass
        elif res[0] == "HU":
            # HU
            # 摸牌 / 和牌 /
            # 其他玩家打牌 / 和牌 /
            pass
        elif res[0] == "PENG":
            # PENG T6
            # 其他玩家打牌 / 已有两张，碰，打出一张 /
            pass
        elif res[0] == "CHI":
            # CHI T2 T5 （T2是顺子的中间牌）
            # 上位玩家出牌 / 已有两张，吃，打出一张 /
            pass
        elif res[0] == "PASS":
            pass
        else:
            raise Exception("Unknown response")

    def step(self):
        currState = self.getState()
        action = self.getAction(currState)
        print(json.dumps({"response": action}))


if __name__ == "__main__":
    mahjong = Mahjong()
    mahjong.step()