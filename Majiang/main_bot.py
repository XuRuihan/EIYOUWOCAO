#!/usr/bin/env python
# encoding: utf-8
'''
@author: zrf
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software:XXXX
@file: main_bot.py
@time: 2020/12/10 13:27
@desc:
'''
import json
import random
from MahjongGB import MahjongFanCalculator
import torch
import torch.optim as optim
from enum import Enum
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

class requests(Enum):
    initialHand = 1
    drawCard = 2
    DRAW = 4
    PLAY = 5
    PENG = 6
    CHI = 7
    GANG = 8
    BUGANG = 9
    MINGGANG = 10
    ANGANG = 11

class responses(Enum):
    PASS = 0
    PLAY = 1
    HU = 2
    # 需要区分明杠和暗杠
    MINGGANG = 3
    ANGANG = 4
    BUGANG = 5
    PENG = 6
    CHI = 7
    need_cards = [0, 1, 0, 0, 1, 1, 0, 1]


class cards(Enum):
    # 饼万条
    B = 0
    W = 9
    T = 18
    # 风
    F = 27
    # 箭牌
    J = 31

class myModel(nn.Module):
    def __init__(self, card_feat_depth, num_extra_feats, num_cards, num_actions):
        super(myModel, self).__init__()
        hidden_channels = [16, 32]
        linear_length = hidden_channels[0] * num_cards * card_feat_depth
        # self.number_card_net = nn.Sequential(
        #     nn.Conv2d(3, hidden_channels[0], 3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.card_net = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.card_decision_net = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[1], 1, (3, 1), stride=1, padding=0),
            nn.Softmax()
        )
        self.action_decision_net = nn.Sequential(
            nn.Linear(num_extra_feats + linear_length, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=0)
        )

    def forward(self, card_feats, device, decide_cards=False, extra_feats=None, action_mask=None, card_mask=None):
        # num_card_feats = np.array([card_feats[:, i*9:i*9+9] for i in range(3)])
        # number_card_feats = torch.from_numpy(num_card_feats).to(device).unsqueeze(0).to(torch.float32)
        # number_card_layer = self.number_card_net(number_card_feats)
        # zi_card_layer = self.zi_card_net(zi_card_feats)
        # print(number_card_layer.shape, zi_card_layer.shape)
        card_feats = torch.from_numpy(card_feats).to(device).unsqueeze(0).unsqueeze(0).to(torch.float32)
        card_layer = self.card_net(card_feats)
        if decide_cards:
            card_mask_tensor = torch.from_numpy(card_mask).to(device)
            card_probs = self.card_decision_net(card_layer).view(-1)
            valid_card_play = self.mask_unavailable_actions(card_probs, card_mask_tensor)
            return valid_card_play
        else:
            action_mask_tensor = torch.from_numpy(action_mask).to(torch.float32).to(device)
            extra_feats_tensor = torch.from_numpy(extra_feats).to(torch.float32).to(device).unsqueeze(0)
            linear_layer = torch.cat((card_layer.view(-1), extra_feats_tensor))
            # print(linear_layer.shape)
            action_probs = self.action_decision_net(linear_layer)
            valid_actions = self.mask_unavailable_actions(action_probs, action_mask_tensor)
            # print(valid_actions, valid_card_play)
            return valid_actions

    def mask_unavailable_actions(self, result, valid_tensor):
        valid_actions = result * valid_tensor
        if valid_tensor.sum() == 0:
            return valid_tensor
        if valid_actions.sum() > 0:
            masked_actions = valid_actions / valid_actions.sum()
        else:
            masked_actions = valid_tensor / valid_tensor.sum()
        return masked_actions

    def train_backward(self, losses, optim):
        loss = torch.cat(losses).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

class MahjongHandler():
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print(self.device)
        self.total_cards = 34
        self.learning_rate = 1e-3
        self.total_actions = len(responses) - 1
        self.model = myModel(3, 1, self.total_cards, self.total_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 40
        self.print_interval = 100
        self.round_count = 0
        self.match = 0
        self.count = 0
        self.reset(True)

    def reset(self, initial=False):
        self.hand_free = np.zeros(34, dtype=int)
        self.hand_fixed = np.zeros(34, dtype=int)
        self.history = np.zeros(34, dtype=int)
        self.fan_count = 0
        self.hand_fixed_data = []
        self.turnID = 0
        self.myPlayerID = 0
        self.quan = 0
        self.prev_request = ''
        self.an_gang_card = ''
        if not initial:
            if self.round_count % self.print_interval == 0:
                print(self.match / self.count)
                self.match = 0
                self.count = 0
            if len(self.loss) > 0:
                self.model.train_backward(self.loss, self.optimizer)
        self.round_count += 1
        self.loss = []

    def step(self, request=None, response_target=None):
        if request is None:
            inputJSON = json.loads(input())
            request = inputJSON['requests'].split(' ')
        else:
            request = request.split(' ')

        request = self.build_hand_history(request)
        if self.turnID <= 1:
            # print(json.dumps({"response": "PASS"}))
            pass
        else:
            available_action_mask, available_card_mask = self.build_available_action_mask(request)
            card_feats = np.array([self.hand_free, self.hand_fixed, self.history])
            extra_feats = np.array(int(available_action_mask[responses.HU.value]==1))
            action_probs = self.model(card_feats, self.device, extra_feats=extra_feats, action_mask=available_action_mask)
            action = int(torch.argmax(action_probs).data.cpu())
            cards = []
            if responses.need_cards.value[action] == 1:
                card_probs = self.model(card_feats, self.device, decide_cards=True, card_mask=available_card_mask[action])
                card_ind = int(torch.argmax(card_probs).data.cpu())
                cards.append(card_ind)
            else:
                card_probs = available_card_mask[action]
                card_ind = np.argmax(card_probs)
            if responses(action) in [responses.PENG, responses.CHI]:
                card_play_probs = self.simulate_chi_peng(request, responses(action), card_feats, card_ind)
                cards.append(int(torch.argmax(card_play_probs).data.cpu()))
            response = self.build_output(responses(action), cards)
            if responses(action) == responses.ANGANG:
                self.an_gang_card = self.getCardName(cards[0])
            # print(json.dumps({"response": response}))

            def judge_response(available_action_mask):
                if available_action_mask.sum() == available_action_mask[responses.PASS.value]:
                    return False
                return True

            if response_target is not None and judge_response(available_action_mask):
                self.loss.extend(self.build_losses(action_probs, request, response_target, card_feats, available_card_mask))
                if len(self.loss) >= self.batch_size:
                    self.model.train_backward(self.loss, self.optimizer)
                    self.loss = []
                self.count += 1
                if response == response_target:
                    # hand = []
                    # for ind, cardcnt in enumerate(self.hand_free):
                    #     for _ in range(cardcnt):
                    #         hand.append(self.getCardName(ind))
                    # print(hand, request, available_action_mask)
                    # print(response, response_target)
                    self.match += 1
        self.prev_request = request
        self.turnID += 1

    def build_losses(self, action_probs, request, response, card_feats, card_mask):
        response = response.split(' ')
        response_name = response[0]
        if response_name == 'GANG':
            if len(response) > 1:
                response_name = 'ANGANG'
                self.an_gang_card = response[-1]
            else:
                response_name = 'MINGGANG'
        response_target = responses[response_name]
        requires_card = responses.need_cards.value[response_target.value]
        action_target = response_target.value
        losses = []

        def build_cross_entropy_loss(probs, target):
            target_tensor = torch.tensor([target]).to(device=self.device, dtype=torch.int64)
            result_tensor = probs.unsqueeze(0)
            loss = F.cross_entropy(result_tensor, target_tensor)
            return loss.unsqueeze(0)

        losses.append(build_cross_entropy_loss(action_probs, action_target))
        if requires_card == 1:
            card_probs = self.model(card_feats, self.device, decide_cards=True,
                                    card_mask=card_mask[responses[response_name].value])
            card_target = self.getCardInd(response[1])
            losses.append(build_cross_entropy_loss(card_probs, card_target))
        if responses[response_name] in [responses.PENG, responses.CHI]:
            if responses[response_name] == responses.CHI:
                chi_peng_ind = self.getCardInd(response[1])
            else:
                chi_peng_ind = self.getCardInd(request[-1])
            card_play_target = self.getCardInd(response[-1])
            card_play_probs = self.simulate_chi_peng(request, responses[response_name], card_feats, chi_peng_ind)
            losses.append(build_cross_entropy_loss(card_play_probs, card_play_target))
        return losses

    def simulate_chi_peng(self, request, response, card_feats, chi_peng_ind):
        last_card_played = self.getCardInd(request[-1])
        available_card_play_mask = np.zeros(self.total_cards)
        if response == responses.CHI:
            card_feats[0][chi_peng_ind - 1:chi_peng_ind + 2] -= 1
            card_feats[0][last_card_played] += 1
            card_feats[2][last_card_played] -= 1
            card_feats[1][chi_peng_ind - 1:chi_peng_ind + 2] += 1
            is_chi = True
        else:
            chi_peng_ind = last_card_played
            card_feats[0][last_card_played] -= 2
            card_feats[2][last_card_played] -= 1
            card_feats[1][last_card_played] += 3
            is_chi = False
        self.build_available_card_mask(available_card_play_mask, responses.PLAY, last_card_played,
                                       chi_peng_ind=chi_peng_ind, is_chi=is_chi)
        card_play_probs = self.model(card_feats, self.device, decide_cards=True, card_mask=available_card_play_mask)
        return card_play_probs

    def build_available_action_mask(self, request):
        available_action_mask = np.zeros(self.total_actions, dtype=int)
        available_card_mask = np.zeros((self.total_actions, self.total_cards), dtype=int)
        requestID = int(request[0])
        playerID = int(request[1])
        myPlayerID = self.myPlayerID
        try:
            last_card = request[-1]
            last_card_ind = self.getCardInd(last_card)
        except:
            last_card = ''
            last_card_ind = 0
        # 摸牌回合
        if requests(requestID) == requests.drawCard:
            for response in [responses.PLAY, responses.ANGANG, responses.BUGANG]:
                self.build_available_card_mask(available_card_mask[response.value], response, last_card_ind, True)
                if available_card_mask[response.value].sum() > 0:
                    available_action_mask[response.value] = 1
            # 杠上开花
            if requests(int(self.prev_request[0])) in [requests.ANGANG, requests.BUGANG]:
                isHu = self.judgeHu(last_card, playerID, True)
            # 这里胡的最后一张牌其实不一定是last_card，因为可能是吃了上家胡，需要知道上家到底打的是哪张
            else:
                isHu = self.judgeHu(last_card, playerID, False)
            if isHu >= 8:
                available_action_mask[responses.HU.value] = 1
                self.fan_count = isHu
        else:
            available_action_mask[responses.PASS.value] = 1
            # 别人出牌
            if requests(requestID) in [requests.PENG, requests.CHI, requests.PLAY]:
                if playerID != myPlayerID:
                    for response in [responses.PENG, responses.MINGGANG, responses.CHI]:
                        self.build_available_card_mask(available_card_mask[response.value], response, last_card_ind, False)
                        if available_card_mask[response.value].sum() > 0:
                            available_action_mask[response.value] = 1
                    # 是你必须现在决定要不要抢胡
                    isHu = self.judgeHu(last_card, playerID, False, dianPao=True)
                    if isHu >= 8:
                        available_action_mask[responses.HU.value] = 1
                        self.fan_count = isHu
            # 抢杠胡
            if requests(requestID) == requests.BUGANG and playerID != myPlayerID:
                isHu = self.judgeHu(last_card, playerID, True, dianPao=True)
                if isHu >= 8:
                    available_action_mask[responses.HU.value] = 1
                    self.fan_count = isHu
        return available_action_mask, available_card_mask

    def build_available_card_mask(self, available_card_mask, response, last_card_ind, chi_peng_ind=None, is_chi=False):
        if response == responses.PLAY:
            # 正常出牌
            if chi_peng_ind is None:
                for i, card_num in enumerate(self.hand_free):
                    if card_num > 0:
                        available_card_mask[i] = 1
            else:
                # 吃了再出
                if is_chi:
                    for i, card_num in enumerate(self.hand_free):
                        if i in [chi_peng_ind - 1, chi_peng_ind, chi_peng_ind + 1] and i != last_card_ind:
                            if card_num > 1:
                                available_card_mask[i] = 1
                        elif card_num > 0:
                            available_card_mask[i] = 1
                else:
                    for i, card_num in enumerate(self.hand_free):
                        if i == chi_peng_ind:
                            if card_num > 2:
                                available_card_mask[i] = 1
                        elif card_num > 0:
                            available_card_mask[i] = 1
        elif response == responses.PENG:
            if self.hand_free[last_card_ind] >= 2:
                available_card_mask[last_card_ind] = 1
        elif response == responses.CHI:
            # 数字牌才可以吃
            if last_card_ind < cards.F.value:
                card_name = self.getCardName(last_card_ind)
                card_number = int(card_name[1])
                for i in [-1, 0, 1]:
                    middle_card = card_number + i
                    if middle_card >= 2 and middle_card <= 8:
                        can_chi = True
                        for card in range(last_card_ind + i - 1, last_card_ind + i + 2):
                            if card != last_card_ind and self.hand_free[card] == 0:
                                can_chi = False
                        if can_chi:
                            available_card_mask[last_card_ind + i] = 1
        elif response == responses.ANGANG:
            for card in range(len(self.hand_free)):
                if self.hand_free[card] == 4:
                    available_card_mask[card] = 1
        elif response == responses.MINGGANG:
            if self.hand_free[last_card_ind] == 3:
                available_card_mask[last_card_ind] = 1
        elif response == responses.BUGANG:
            for card in range(len(self.hand_free)):
                if self.hand_fixed[card] == 3 and self.hand_free[card] == 1:
                    available_card_mask[card] = 1
        else:
            available_card_mask[last_card_ind] = 1
        return available_card_mask

    def judgeHu(self, last_card, playerID, isGANG, isJUEZHANG=False, isLAST=False, dianPao=False):
        hand = []
        for ind, cardcnt in enumerate(self.hand_free):
            for _ in range(cardcnt):
                hand.append(self.getCardName(ind))
        if not dianPao:
            hand.remove(last_card)
        try:
            ans = MahjongFanCalculator(tuple(self.hand_fixed_data), tuple(hand), last_card, 0, playerID==self.myPlayerID,
                                       isJUEZHANG, isGANG, isLAST, self.myPlayerID, self.quan)
        except Exception as err:
            # print(hand, last_card, self.hand_fixed_data)
            # print(err)
            if str(err) == 'ERROR_NOT_WIN':
                return 0
            else:
                print(hand, last_card, self.hand_fixed_data)
                print(err)
                return 0
        else:
            fan_count = 0
            for fan in ans:
                fan_count += fan[0]
            return fan_count

    def build_hand_history(self, request):
        # 第0轮，确定位置
        if self.turnID == 0:
            _, myPlayerID, quan = request
            self.myPlayerID = int(myPlayerID)
            self.quan = int(quan)
            return request
        # 第一轮，发牌
        if self.turnID == 1:
            for i in range(5, 18):
                cardInd = self.getCardInd(request[i])
                self.hand_free[cardInd] += 1
            return request
        if int(request[0]) == 3:
            request[0] = str(requests[request[2]].value)
        elif int(request[0]) == 2:
            request.insert(1, self.myPlayerID)
        requestID = int(request[0])
        playerID = int(request[1])
        myPlayerID = self.myPlayerID
        if requests(requestID) == requests.drawCard:
            self.hand_free[self.getCardInd(request[-1])] += 1
        elif requests(requestID) == requests.PLAY:
            self.history[self.getCardInd(request[-1])] += 1
            if playerID == myPlayerID:
                self.hand_free[self.getCardInd(request[-1])] -= 1
        elif requests(requestID) == requests.PENG:
            # 上一步一定有play
            last_card_ind = self.getCardInd(self.prev_request[-1])
            play_card_ind = self.getCardInd(request[-1])
            if playerID != myPlayerID:
                self.history[last_card_ind] += 2
                self.history[play_card_ind] += 1
            else:
                # 记录peng来源于哪个玩家
                if int(self.prev_request[1]) - myPlayerID < 0:
                    self.hand_fixed_data.append(('PENG', self.prev_request[-1], int(self.prev_request[1]) - myPlayerID + 4))
                else:
                    self.hand_fixed_data.append(('PENG', self.prev_request[-1], int(self.prev_request[1]) - myPlayerID))
                self.hand_free[last_card_ind] -= 2
                self.hand_fixed[last_card_ind] += 3
                self.history[last_card_ind] -= 1
                self.history[play_card_ind] += 1
                self.hand_free[play_card_ind] -= 1
        elif requests(requestID) == requests.CHI:
            # 上一步一定有play
            last_card_ind = self.getCardInd(self.prev_request[-1])
            middle_card, play_card = request[3:5]
            middle_card_ind = self.getCardInd(middle_card)
            play_card_ind = self.getCardInd(play_card)
            if playerID != myPlayerID:
                self.history[middle_card_ind-1:middle_card_ind+2] += 1
                self.history[last_card_ind] -= 1
                self.history[play_card_ind] += 1
            else:
                # CHI,中间牌名，123代表上家的牌是第几张
                self.hand_fixed_data.append(('CHI', middle_card, last_card_ind - middle_card_ind + 2))
                self.hand_free[middle_card_ind-1:middle_card_ind+2] -= 1
                self.hand_free[last_card_ind] += 1
                self.hand_fixed[middle_card_ind-1:middle_card_ind+2] += 1
                self.history[last_card_ind] -= 1
                self.history[play_card_ind] += 1
                self.hand_free[play_card_ind] -= 1
        elif requests(requestID) == requests.GANG:
            # 暗杠
            if requests(int(self.prev_request[0])) in [requests.drawCard, requests.DRAW]:
                request[2] = requests.ANGANG.name
                if playerID == myPlayerID:
                    gangCard = self.an_gang_card
                    print(gangCard)
                    gangCardInd = self.getCardInd(gangCard)
                    # 记录gang来源于哪个玩家（可能来自自己，暗杠）
                    self.hand_fixed_data.append(('GANG', gangCard, 0))
                    self.hand_fixed[gangCardInd] = 4
                    self.hand_free[gangCardInd] = 0
                    self.history[gangCardInd] = 0
            else:
                # 明杠
                gangCardInd = self.getCardInd(self.prev_request[-1])
                request[2] = requests.ANGANG.name
                if playerID != myPlayerID:
                    self.history[gangCardInd] = 4
                else:
                    # 记录gang来源于哪个玩家
                    if int(self.prev_request[1]) - myPlayerID < 0:
                        self.hand_fixed_data.append(
                            ('GANG', self.prev_request[-1], int(self.prev_request[1]) - myPlayerID + 4))
                    else:
                        self.hand_fixed_data.append(
                            ('GANG', self.prev_request[-1], int(self.prev_request[1]) - myPlayerID))
                    self.hand_fixed[gangCardInd] = 4
                    self.hand_free[gangCardInd] = 0
                    self.history[gangCardInd] = 0
        elif requests(requestID) == requests.BUGANG:
            # 补杠没有在番数计算吗？？？？？
            play_card_ind = self.getCardInd(request[-1])
            self.history[play_card_ind] += 1
            if playerID == myPlayerID:
                self.hand_free[play_card_ind] -= 1
                self.hand_fixed[play_card_ind] += 1
        return request

    def build_output(self, response, cards_ind):
        if responses.need_cards.value[response.value] == 1 or response == responses.PENG:
            response_name = response.name
            if response == responses.ANGANG:
                response_name = 'GANG'
            return '{} {}'.format(response_name, self.getCardName(cards_ind[0]))
        if response == responses.CHI:
            return 'CHI {} {}'.format(self.getCardName(cards_ind[0]), self.getCardName(cards_ind[1]))
        response_name = response.name
        if response == responses.MINGGANG:
            response_name = 'GANG'
        return response_name


    def getCardInd(self, cardName):
        return cards[cardName[0]].value + int(cardName[1]) - 1

    def getCardName(self, cardInd):
        num = 1
        while True:
            if cardInd in cards._value2member_map_:
                break
            num += 1
            cardInd -= 1
        return cards(cardInd).name + str(num)

actions = []
def build_input(folder):
    dirs = os.listdir(folder)
    for dir in dirs:
        subfolder = folder + '/' + dir
        files = os.listdir(subfolder)
        for i, file in enumerate(files):
            if i % 100 == 0:
                print('{}/{}'.format(i, len(files)))
            with open(subfolder + '/' + file, encoding='utf-8') as f:
                line = f.readline()
                count = 1
                while line:
                    line = line.strip('\n').split('\t')
                    if count >= 7:
                        act = line[1]
                        if act not in actions:
                            actions.append(act)
                    line = f.readline()
                    count += 1

my_bot = MahjongHandler()
# build_input('C:\\Users\\zrf19\\Desktop\\强化学习\\麻将\\mjdata\\output2017')
# print(actions)
inputs = {
	"requests": [
		"0 3 3",
		"1 0 0 0 0 W1 B1 T3 W4 B1 T8 F1 J1 W5 B3 W8 F3 W9",
		"3 0 DRAW",
		"3 0 PLAY F2",
		"3 1 DRAW",
		"3 1 PLAY F1",
		"3 2 DRAW",
		"3 2 PLAY F4",
		"2 T1",
		"3 3 PLAY F1",
		"3 0 DRAW",
		"3 0 PLAY J1",
		"3 1 DRAW",
		"3 1 PLAY F4",
		"3 2 DRAW",
		"3 2 PLAY W8",
		"2 F4",
		"3 3 PLAY F4",
		"3 0 DRAW",
		"3 0 PLAY F3",
		"3 1 DRAW",
		"3 1 PLAY J3",
		"3 2 DRAW",
		"3 2 PLAY W9",
		"2 T9",
		"3 3 PLAY F3",
		"3 0 DRAW",
		"3 0 PLAY W9",
		"3 1 DRAW",
		"3 1 PLAY J3",
		"3 2 DRAW",
		"3 2 PLAY W6",
		"3 3 CHI W5 J1",
		"3 0 DRAW",
		"3 0 PLAY F2",
		"3 1 DRAW",
		"3 1 PLAY W9",
		"3 2 DRAW",
		"3 2 PLAY F4",
		"2 T2",
		"3 3 PLAY W9",
		"3 0 DRAW",
		"3 0 PLAY T3",
		"3 1 DRAW",
		"3 1 PLAY B8",
		"3 2 DRAW",
		"3 2 PLAY T7",
		"3 3 CHI T8 B1",
		"3 0 DRAW",
		"3 0 PLAY B3",
		"3 1 DRAW",
		"3 1 PLAY W2",
		"3 2 DRAW",
		"3 2 PLAY B1",
		"2 W2",
		"3 3 PLAY W8",
		"3 0 DRAW",
		"3 0 PLAY B4",
		"3 1 DRAW",
		"3 1 PLAY B5",
		"3 2 DRAW",
		"3 2 PLAY B6",
		"2 B6",
		"3 3 PLAY W2",
		"3 0 CHI W2 W4",
		"3 1 DRAW",
		"3 1 PLAY F3",
		"3 2 DRAW",
		"3 2 PLAY W4",
		"2 T4",
		"3 3 PLAY W1",
		"3 0 DRAW",
		"3 0 PLAY J2",
		"3 1 DRAW",
		"3 1 PLAY B5",
		"3 2 DRAW",
		"3 2 PLAY W8",
		"2 W4",
		"3 3 PLAY W4",
		"3 0 DRAW",
		"3 0 PLAY B5",
		"3 1 DRAW",
		"3 1 PLAY W1",
		"3 2 DRAW",
		"3 2 PLAY T9"
	],
	"responses": [
		"PASS",
		"PASS",
		"PASS",
		"PASS"
	]
}
print(torch.__version__)
print(torch.cuda.is_available())
for i in range(8):
    with open('Tread {}-mini.json'.format(i), 'r') as f:
        rounds_data = json.load(f)
        for round_data in rounds_data:
            for i in range(4):
                train_requests = round_data["requests"][i]
                first_request = '0 {} {}'.format(i, round_data['zhuangjia'])
                train_requests.insert(0, first_request)
                train_responses = ['PASS'] + round_data["responses"][i]
                for _request, _response in zip(train_requests, train_responses):
                    # print(_request, _response, round_data['fname'])
                    my_bot.step(_request, _response)
                my_bot.reset()
    # print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
    # sys.stdout.flush()
# stmp = ''
# json_str = input()
# inputJSON = json.loads(json_str)
# turnID = len(inputJSON["responses"])
# request = []
# response = []
# for i in range(turnID):
#     request.append(str(inputJSON["requests"][i]))
#     response.append(str(inputJSON["responses"][i]))
#
# request.append(str(inputJSON["requests"][turnID]))
#
# if turnID < 2:
#     response.append('PASS')
# else:
#     itmp, myPlayerID, quan = request[0].split(' ')
#     hand = []
#     req = request[1]
#     req = req.split(' ')
#     for i in range(5, 18):
#         hand.append(req[i])
#     for i in range(2, turnID):
#         req = request[i].split(' ')
#         res = response[i].split(' ')
#         if req[0] == '2':
#             hand.append(req[1])
#             hand.remove(res[1])
#     req = request[turnID].split(' ')
#     if req[0] == '2':
#         random.shuffle(hand)
#         resp = 'PLAY ' + hand[-1]
#         hand.pop()
#     else:
#         resp = 'PASS'
#     response.append(resp)
# print(json.dumps({"response": response[turnID]}))



