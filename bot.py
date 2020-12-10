import json
import random
from MahjongGB import MahjongFanCalculator

"""某个运行实例的输入。每行代表一个step
{"requests":["0 0 3"],"responses":[]}
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2"],"responses":["PASS"]}
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2","2 F2"],"responses":["PASS","PASS"]}
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2","2 F2","3 0 PLAY F1"],"responses":["PASS","PASS","PLAY F1"]}
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2","2 F2","3 0 PLAY F1","3 1 DRAW"],"responses":["PASS","PASS","PLAY F1","PASS"]}
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2","2 F2","3 0 PLAY F1","3 1 DRAW","3 1 PLAY T4"],"responses":["PASS","PASS","PLAY F1","PASS","PASS"]}
"""

# 算番函数
'''
((value,descripthon),...) MahjongFanCalculator(
    pack=((packType,tileCode,data),...),
    hand=(tileCode,...),
    winTile,
    flowerCount,
    isZIMO,
    isJUEZHANG,
    isGANG,
    isLAST,
    menFeng,
    quanFeng)
'''

# 定义 request 常量
DRAW_AND_PLAY = "2"


class Mahjong():
    def __init__(self):
        self.turnID = -1
        self.playerID = 0
        self.quan = 0
        self.hand = []

    def reset(self):  # 用于后期训练 RL 模型的接口
        self.turnID = -1
        self.playerID = 0
        self.quan = 0
        self.hand = []

    def getState(self):  # 读取数据并转换成当前状态
        inputJson = json.loads(input())
        requests, responses = inputJson["requests"], inputJson["responses"]
        self.turnID = len(responses)
        if self.turnID == 0:  # 未知作用
            _, self.playerID, self.quan = tuple(map(int, requests[0].split()))
        elif self.turnID == 1:  # 起始手牌，只有在跨回合运行的bot中才能用上
            self.hand = requests[1].split()[5:18]
        else:
            self.hand = requests[1].split()[5:18]
            for i in range(2, self.turnID):
                req = requests[i].split()
                res = responses[i].split()
                if req[0] == DRAW_AND_PLAY:  # 摸牌
                    self.hand.append(req[1])
                    self.hand.remove(res[1])
        req = requests[-1].split()  # 当前回合输入
        return req

    def getAction(self, req):
        if self.turnID < 2:
            return "PASS"
        if req[0] == DRAW_AND_PLAY:
            playCard = random.choice(self.hand)
            return "PLAY " + playCard
        else:
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
        state = self.getState()
        action = self.getAction(state)
        print(json.dumps({"response": action}))


if __name__ == "__main__":
    mahjong = Mahjong()
    mahjong.step()