import json
import random
import torch
from const import *  # 导入常量
from MahjongGB import MahjongFanCalculator  # 算番
"""某个运行实例的输入。每行代表一个step
{"requests":["0 0 3","1 0 0 0 0 B2 B1 B1 J2 J3 W2 T5 W2 F1 T3 F4 T8 F2","2 F2","3 0 PLAY F1","3 1 DRAW","3 1 PLAY T4"],"responses":["PASS","PASS","PLAY F1","PASS","PASS"]}
"""


class Mahjong():
    def __init__(self):
        self.reset()

    def reset(self):  # 用于后期训练 RL 模型的接口
        self.turnID = -1
        self.menFeng = 0  # playerID 就是门风
        self.quanFeng = 0

        self.pack = []  # ((packType, tileCode, data), ...),
        self.hand = [0] * TILE_NUM

        self.unknown = [4] * TILE_NUM  # 记录已经打出的牌
        self.left = 21  # 初始牌墙中的牌数

    # 字符串表示转换成向量表示
    def deck2vector(self):
        # TODO
        pass

    # 向量表示转换成字符串表示
    def vector2deck(self):
        # TODO
        pass

    def calculateFan(self, tileCode, isZIMO, isGANG):
        handcode = []
        for idx, num in enumerate(self.hand):
            handcode += [idx2tile[idx]] * num
        isJUEZHANG = (self.unknown[tile2idx[tileCode]] == 1)
        isLAST = (self.left == 0)

        try:
            fans = MahjongFanCalculator(
                self.pack,  # 明牌
                handcode,  # 暗牌
                tileCode,  # 和的那张牌
                0,  # 补花数，在本模型中应当设为 0
                isZIMO,  # 自摸
                isJUEZHANG,  # 绝张
                isGANG,  # 抢杠和/杠上开花
                isLAST,  # 牌墙最后一张，妙手回春/海底捞月
                self.menFeng,  # 门风
                self.quanFeng,  # 圈风
            )
            totalFan = sum(fan[0] for fan in fans)  # fan[0] 为番数，fan[1] 为番形名称
            return totalFan
        except Exception as e:
            return 0  # 没有和，番数为 0

    def tileFrom(self, playerID):
        # 0 - 本家，1 - 上家，2 - 对家，3 - 下家
        return (self.menFeng - playerID) % 4

    def reviewOneStep(self, req, last_req):
        '''一步复盘
        Args:
            req - 本回合 botzone 发给所有玩家的信息
            last_req - 上回合 botzone 发给所有玩家的信息
        Return:
            无返回值
        '''
        curr_idx = tile2idx.get(req[-1], None)
        last_idx = tile2idx.get(last_req[-1], None)
        if req[0] == SELF_DRAW:  # 当前为己方玩家摸牌
            self.left -= 1
            self.hand[curr_idx] += 1
            self.unknown[curr_idx] -= 1

        elif req[0] == OTHERS_TURN:  # 当前不为己方玩家摸牌
            playerID = int(req[1])  # 当前玩家编号
            if req[2] == DRAW:  # 摸牌，此时玩家编号不可能是自己
                pass

            elif req[2] == PLAY:  # 打牌
                if playerID == self.menFeng:
                    self.hand[curr_idx] -= 1
                else:
                    self.unknown[curr_idx] -= 1

            elif req[2] == PENG:  # 碰
                if playerID == self.menFeng:
                    self.hand[last_idx] -= 2
                    self.hand[curr_idx] -= 1
                    self.pack.append(
                        (PENG, idx2tile[last_idx], self.tileFrom(playerID)))
                else:
                    self.unknown[last_idx] -= 2  # 上回合打出的牌
                    self.unknown[curr_idx] -= 1  # 碰之后打出的牌

            elif req[2] == CHI:  # 吃
                mididx = tile2idx[req[-2]]
                chi = {mididx - 1, mididx, mididx + 1}
                chi.remove(last_idx)
                if playerID == self.menFeng:
                    for idx in chi:
                        self.hand[idx] -= 1
                    self.hand[curr_idx] -= 1
                    self.pack.append(
                        (CHI, idx2tile[mididx], last_idx - mididx + 2))
                else:
                    for idx in chi:
                        self.unknown[idx] -= 1  # 吃的另外两张牌
                    self.unknown[curr_idx] -= 1  # 吃之后打出的牌

            elif req[2] == GANG:  # 杠
                if playerID == self.menFeng:  # 如果是自己，那么明杠暗杠都会少手牌
                    self.hand[last_idx] -= 3
                    self.pack.append(
                        (GANG, idx2tile[last_idx], self.tileFrom(playerID)))
                # 他人明杠可以用于计算绝张
                elif last_req[-1] != DRAW and last_req[0] != SELF_DRAW:
                    self.unknown[last_idx] -= 3

            elif req[2] == BUGANG:  # 补杠
                if playerID == self.menFeng:
                    self.hand[curr_idx] -= 1
                    self.pack.append(
                        (GANG, idx2tile[last_idx], self.tileFrom(playerID)))
                else:
                    self.unknown[curr_idx] -= 1

    def review(self):
        '''完整复盘
        Args:
            （命令行输入）
        Return:
            req - 当前输入的 request 切分的字符串列表
            last_req - 上一次输入的 request 切分的字符串列表
        '''
        # inputJson = json.loads(input())
        with open("input1.json", "r", encoding="utf8") as f:
            inputJson = json.load(f)
        requests = inputJson["requests"]
        self.turnID = len(requests) - 1

        last_req = requests[0].split()  # 用于记录上一步输入内容
        if self.turnID >= 2:
            # 记录 menFeng 和 quanFeng（playerID 就是门风）
            _, self.menFeng, self.quanFeng = list(map(int,
                                                      requests[0].split()))

            # 起始手牌，只有在跨回合运行的bot中才能用上
            handcode = requests[1].split()[5:18]
            for tile in handcode:
                self.hand[tile2idx[tile]] += 1
                self.unknown[tile2idx[tile]] -= 1

            # 从第2回合开始，根据每一步输入输出恢复状态
            for i in range(2, self.turnID):
                req = requests[i].split()
                self.reviewOneStep(req, last_req)
                last_req = req

        req = requests[-1].split()  # 当前回合输入
        return req

    def getAction(self, req):
        if self.turnID < 2:
            return PASS
        if req[0] == SELF_DRAW:  # 自摸
            # 这里需要调整策略
            # 此时可行的策略为【和/打牌/暗杠/补杠】
            # isGANG 的判断需要改写
            fan = self.calculateFan(req[-1], isZIMO=True, isGANG=False)
            if fan >= QIHU:
                return HU
            playtile = random.choice(self.hand)
            return f"PLAY {playtile}"

        elif req[0] == OTHERS_TURN:  # 其他家出牌
            playerID = int(req[1])
            # 此时如果是【他人摸牌/杠（暗杠）/补杠/自己打牌】，则无需任何操作
            if req[2] == DRAW or req[2] == GANG or req[2] == BUGANG \
                or (req[2] == PLAY and playerID == self.menFeng):
                return PASS

            # 否则可行的策略为【和/碰/杠/吃（需要判断上家）】
            else:
                if self.tileFrom(playerID) == 1:  # 来自上家
                    pass
                return PASS

    def step(self):
        req = self.review()  # 复盘之前的状态
        action = self.getAction(req)
        print(json.dumps({"response": action}))


if __name__ == "__main__":
    mahjong = Mahjong()
    mahjong.step()