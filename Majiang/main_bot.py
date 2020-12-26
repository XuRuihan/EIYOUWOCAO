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
from MahjongGB import MahjongFanCalculator
import torch
import torch.optim as optim
from enum import Enum
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random

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
    loss_weight = [1, 2, 20, 5, 5, 5, 3, 3]

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
        hidden_channels = [8, 16, 32]
        hidden_layers_size = [512, 1024]
        linear_length = hidden_channels[1] * num_cards * card_feat_depth
        # self.number_card_net = nn.Sequential(
        #     nn.Conv2d(3, hidden_channels[0], 3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.card_net = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.card_decision_net = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[2], 1, (card_feat_depth, 1), stride=1, padding=0)
        )
        self.action_decision_net = nn.Sequential(
            nn.Linear(num_extra_feats + linear_length, hidden_layers_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[0], hidden_layers_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[1], num_actions)
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
            card_mask_tensor = torch.from_numpy(card_mask).to(torch.float32).to(device)
            card_probs = self.card_decision_net(card_layer).view(-1)
            valid_card_play = self.mask_unavailable_actions(card_probs, card_mask_tensor)
            return valid_card_play
        else:
            action_mask_tensor = torch.from_numpy(action_mask).to(torch.float32).to(device)
            extra_feats_tensor = torch.from_numpy(extra_feats).to(torch.float32).to(device)
            linear_layer = torch.cat((card_layer.view(-1), extra_feats_tensor))
            # print(linear_layer.shape)
            action_probs = self.action_decision_net(linear_layer)
            valid_actions = self.mask_unavailable_actions(action_probs, action_mask_tensor)
            # print(valid_actions, valid_card_play)
            return valid_actions

    def mask_unavailable_actions(self, result, valid_tensor):
        valid_actions = result * valid_tensor
        return valid_actions

    def train_backward(self, losses, optim):
        loss = torch.cat(losses).mean()
        # print(loss)
        # if loss.requires_grad is False:
        #     print(losses)
        optim.zero_grad()
        loss.backward()
        optim.step()

class MahjongHandler():
    def __init__(self, train, model_path, load_model=False, save_model=True, botzone=False):
        use_cuda = torch.cuda.is_available()
        self.botzone = botzone
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if not botzone:
            print('using ' + str(self.device))
        self.train = train
        self.model_path = model_path
        self.load_model = load_model
        self.save_model = save_model
        self.total_cards = 34
        self.learning_rate = 1e-4
        self.action_loss_weight = responses.loss_weight.value
        self.card_loss_weight = 2
        self.total_actions = len(responses) - 2
        self.model = myModel(

                        card_feat_depth=18,
                        num_extra_feats=self.total_actions,
                        num_cards=self.total_cards,
                        num_actions=self.total_actions
                        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_precision = 0
        self.batch_size = 500
        self.print_interval = 100
        self.save_interval = 500
        self.round_count = 0
        self.match = np.zeros(self.total_actions)
        self.count = np.zeros(self.total_actions)
        if self.load_model:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                self.round_count = checkpoint['progress']
            except:
                pass
            # state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            # torch.save(state, self.model_path, _use_new_zipfile_serialization=False)
        if not train:
            self.model.eval()
        self.reset(True)

    def reset(self, initial=False):
        self.hand_free = np.zeros(self.total_cards, dtype=int)
        self.history = np.zeros(self.total_cards, dtype=int)
        self.player_history = np.zeros((4, self.total_cards), dtype=int)
        self.player_on_table = np.zeros((4, self.total_cards), dtype=int)
        self.hand_fixed = self.player_on_table[0]
        self.player_last_play = np.zeros(4, dtype=int)
        self.player_angang = np.zeros(4, dtype=int)
        self.fan_count = 0
        self.hand_fixed_data = []
        self.turnID = 0
        self.tile_count = [21, 21, 21, 21]
        self.myPlayerID = 0
        self.quan = 0
        self.prev_request = ''
        self.an_gang_card = ''
        if not initial:
            if len(self.loss) > 0:
                self.model.train_backward(self.loss, self.optimizer)
            if not self.botzone:
                if self.round_count % self.print_interval == 0:
                    precisions = self.match / self.count
                    for i in range(self.total_actions):
                        print('{}: {}/{} {:2%}'.format(responses(i).name, self.match[i], self.count[i], precisions[i]))
                    self.match = np.zeros(self.total_actions)
                    self.count = np.zeros(self.total_actions)
                if self.round_count % self.save_interval == 0:
                    state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'progress': self.round_count}
                    torch.save(state, self.model_path, _use_new_zipfile_serialization=False)
        self.round_count += 1
        self.loss = []

    def step(self, request=None, response_target=None, fname=None):
        if fname:
            self.fname = fname
        if request is None:
            if self.turnID == 0:
                inputJSON = json.loads(input())
                request = inputJSON['requests'][0].split(' ')
            else:
                request = input().split(' ')
        else:
            request = request.split(' ')

        request = self.build_hand_history(request)
        if self.turnID <= 1:
            if self.botzone:
                print(json.dumps({"response": "PASS"}))
        else:
            available_action_mask, available_card_mask = self.build_available_action_mask(request)
            card_feats = self.build_input(self.hand_free, self.history, self.player_history,
                                   self.player_on_table, self.player_last_play, self.player_angang)
            extra_feats = available_action_mask
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
                card_play_probs = self.simulate_chi_peng(request, responses(action), card_ind)
                cards.append(int(torch.argmax(card_play_probs).data.cpu()))
            response = self.build_output(responses(action), cards)
            if responses(action) == responses.ANGANG:
                self.an_gang_card = self.getCardName(cards[0])
            if self.botzone:
                print(json.dumps({"response": response}))

            def judge_response(available_action_mask):
                if available_action_mask.sum() == available_action_mask[responses.PASS.value]:
                    return False
                return True

            if self.train and response_target is not None and judge_response(available_action_mask):
                self.loss.extend(self.build_losses(action_probs, request, response_target, card_feats,
                                                   available_card_mask, available_action_mask))
                if len(self.loss) >= self.batch_size:
                    self.model.train_backward(self.loss, self.optimizer)
                    self.loss = []
                self.build_result_summary(response, response_target)

        self.prev_request = request
        self.turnID += 1

    def build_input(self, my_free, history, play_history, on_table, last_play, angang):
        temp = np.array([my_free, 4 - history])
        # total_cards = my_free + on_table[0]
        # if total_cards.sum() < 13 or total_cards.sum() > 14:
        #     print('wwwww', total_cards)
        # print(play_history.sum(axis=0))
        # cards_shown = 4 - history + my_free + on_table.sum(axis=0) + play_history.sum(axis=0)
        # if cards_shown.max() > 5:
        #     print(cards_shown)
        #     print(history)
        #     print(my_free)
        #     print(on_table.sum(0))
        #     print(play_history.sum(0))
        # print(cards_shown)
        one_hot_angang = np.eye(self.total_cards)[angang]
        one_hot_last_play = np.eye(self.total_cards)[last_play]
        card_feats = np.concatenate((temp, on_table, play_history, one_hot_last_play, one_hot_angang))
        return card_feats

    def build_result_summary(self, response, response_target):
        resp_name = response.split(' ')[0]
        resp_target_name = response_target.split(' ')[0]
        if resp_target_name == 'GANG':
            if len(response_target.split(' ')) > 1:
                resp_target_name = 'ANGANG'
            else:
                resp_target_name = 'MINGGANG'
        if resp_name == 'GANG':
            if len(response.split(' ')) > 1:
                resp_name = 'ANGANG'
            else:
                resp_name = 'MINGGANG'
        self.count[responses[resp_target_name].value] += 1
        if response == response_target:
            self.match[responses[resp_name].value] += 1

    def translate_hand(self):
        hand = []
        for ind, cardcnt in enumerate(self.hand_free):
            for _ in range(cardcnt):
                hand.append(self.getCardName(ind))
        print(hand, self.hand_fixed_data)

    def build_losses(self, action_probs, request, response, card_feats, card_mask, action_mask):
        response = response.split(' ')
        response_name = response[0]
        if response_name == 'GANG':
            if len(response) > 1:
                response_name = 'ANGANG'
                self.an_gang_card = response[-1]
            else:
                response_name = 'MINGGANG'
        if action_mask[responses[response_name].value] == 0:
            self.translate_hand()
            print(response_name, self.fname)
        response_target = responses[response_name]
        requires_card = responses.need_cards.value[response_target.value]
        action_target = response_target.value
        losses = []

        def build_cross_entropy_loss(probs, target, weight):
            target_tensor = torch.tensor([target]).to(device=self.device, dtype=torch.int64)
            result_tensor = probs.unsqueeze(0)
            loss = F.cross_entropy(result_tensor, target_tensor).to(torch.float32) * weight
            # if loss.requires_grad == False:
            #     print(probs, target)
            return loss.unsqueeze(0)

        action_weight = self.action_loss_weight[responses[response_name].value]
        losses.append(build_cross_entropy_loss(action_probs, action_target, action_weight))
        if requires_card == 1:
            card_probs = self.model(card_feats, self.device, decide_cards=True,
                                    card_mask=card_mask[responses[response_name].value])
            card_target = self.getCardInd(response[1])
            losses.append(build_cross_entropy_loss(card_probs, card_target, self.card_loss_weight))
        if responses[response_name] in [responses.PENG, responses.CHI]:
            if responses[response_name] == responses.CHI:
                chi_peng_ind = self.getCardInd(response[1])
            else:
                chi_peng_ind = self.getCardInd(request[-1])
            card_play_target = self.getCardInd(response[-1])
            card_play_probs = self.simulate_chi_peng(request, responses[response_name], chi_peng_ind)
            losses.append(build_cross_entropy_loss(card_play_probs, card_play_target, self.card_loss_weight))
        return losses

    def simulate_chi_peng(self, request, response, chi_peng_ind):
        last_card_played = self.getCardInd(request[-1])
        available_card_play_mask = np.zeros(self.total_cards)
        my_free, on_table = self.hand_free.copy(), self.player_on_table.copy()
        if response == responses.CHI:
            my_free[chi_peng_ind - 1:chi_peng_ind + 2] -= 1
            my_free[last_card_played] += 1
            on_table[0][chi_peng_ind - 1:chi_peng_ind + 2] += 1
            is_chi = True
        else:
            chi_peng_ind = last_card_played
            my_free[last_card_played] -= 2
            on_table[0][last_card_played] += 3
            is_chi = False
        self.build_available_card_mask(available_card_play_mask, responses.PLAY, last_card_played,
                                       chi_peng_ind=chi_peng_ind, is_chi=is_chi)
        card_feats = self.build_input(my_free, self.history, self.player_history, on_table, self.player_last_play,
                                      self.player_angang)
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
                        # 不是上家
                        if response == responses.CHI and (self.myPlayerID - playerID) % 4 != 1:
                            continue
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

    def judgeHu(self, last_card, playerID, isGANG, dianPao=False):
        hand = []
        for ind, cardcnt in enumerate(self.hand_free):
            for _ in range(cardcnt):
                hand.append(self.getCardName(ind))
        if self.history[self.getCardInd(last_card)] == 4:
            isJUEZHANG = True
        else:
            isJUEZHANG = False
        if sum(self.tile_count) == 0:
            isLAST = True
        else:
            isLAST = False
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
            self.other_players_id = [(self.myPlayerID - i) % 4 for i in range(4)]
            self.player_positions = {}
            for position, id in enumerate(self.other_players_id):
                self.player_positions[id] = position
            self.quan = int(quan)
            return request
        # 第一轮，发牌
        if self.turnID == 1:
            for i in range(5, 18):
                cardInd = self.getCardInd(request[i])
                self.hand_free[cardInd] += 1
                self.history[cardInd] += 1
            return request
        if int(request[0]) == 3:
            request[0] = str(requests[request[2]].value)
        elif int(request[0]) == 2:
            request.insert(1, self.myPlayerID)
        request = self.maintain_status(request, self.hand_free, self.history, self.player_history,
                                   self.player_on_table, self.player_last_play, self.player_angang)
        # if requests(requestID) in [requests.drawCard, requests.DRAW]:
        #     self.tile_count[playerID] -= 1
        # if requests(requestID) == requests.drawCard:
        #     self.hand_free[self.getCardInd(request[-1])] += 1
        # elif requests(requestID) == requests.PLAY:
        #     self.history[self.getCardInd(request[-1])] += 1
        #     self.player_history[player_position] += 1
        #     if playerID == myPlayerID:
        #         self.hand_free[self.getCardInd(request[-1])] -= 1
        # elif requests(requestID) == requests.PENG:
        #     # 上一步一定有play
        #     last_card_ind = self.getCardInd(self.prev_request[-1])
        #     play_card_ind = self.getCardInd(request[-1])
        #     self.player_on_table[player_position]
        #     if playerID != myPlayerID:
        #         self.history[last_card_ind] += 2
        #         self.history[play_card_ind] += 1
        #     else:
        #         # 记录peng来源于哪个玩家
        #         last_player = int(self.prev_request[1])
        #         self.hand_fixed_data.append(('PENG', self.prev_request[-1], (last_player - myPlayerID) % 4))
        #         self.hand_free[last_card_ind] -= 2
        #         self.hand_fixed[last_card_ind] += 3
        #         self.history[last_card_ind] -= 1
        #         self.history[play_card_ind] += 1
        #         self.hand_free[play_card_ind] -= 1
        # elif requests(requestID) == requests.CHI:
        #     # 上一步一定有play
        #     last_card_ind = self.getCardInd(self.prev_request[-1])
        #     middle_card, play_card = request[3:5]
        #     middle_card_ind = self.getCardInd(middle_card)
        #     play_card_ind = self.getCardInd(play_card)
        #     if playerID != myPlayerID:
        #         self.history[middle_card_ind-1:middle_card_ind+2] += 1
        #         self.history[last_card_ind] -= 1
        #         self.history[play_card_ind] += 1
        #     else:
        #         # CHI,中间牌名，123代表上家的牌是第几张
        #         self.hand_fixed_data.append(('CHI', middle_card, last_card_ind - middle_card_ind + 2))
        #         self.hand_free[middle_card_ind-1:middle_card_ind+2] -= 1
        #         self.hand_free[last_card_ind] += 1
        #         self.hand_fixed[middle_card_ind-1:middle_card_ind+2] += 1
        #         self.history[last_card_ind] -= 1
        #         self.history[play_card_ind] += 1
        #         self.hand_free[play_card_ind] -= 1
        # elif requests(requestID) == requests.GANG:
        #     # 暗杠
        #     if requests(int(self.prev_request[0])) in [requests.drawCard, requests.DRAW]:
        #         request[2] = requests.ANGANG.name
        #         if playerID == myPlayerID:
        #             gangCard = self.an_gang_card
        #             # print(gangCard)
        #             if gangCard == '' and not self.botzone:
        #                 print(self.prev_request)
        #                 print(request)
        #                 print(self.fname)
        #             gangCardInd = self.getCardInd(gangCard)
        #             # 记录gang来源于哪个玩家（可能来自自己，暗杠）
        #             self.hand_fixed_data.append(('GANG', gangCard, 0))
        #             self.hand_fixed[gangCardInd] = 4
        #             self.hand_free[gangCardInd] = 0
        #             self.history[gangCardInd] = 0
        #     else:
        #         # 明杠
        #         gangCardInd = self.getCardInd(self.prev_request[-1])
        #         request[2] = requests.ANGANG.name
        #         if playerID != myPlayerID:
        #             self.history[gangCardInd] = 4
        #         else:
        #             # 记录gang来源于哪个玩家
        #             last_player = int(self.prev_request[1])
        #             self.hand_fixed_data.append(
        #                     ('GANG', self.prev_request[-1], (last_player - myPlayerID) % 4))
        #             self.hand_fixed[gangCardInd] = 4
        #             self.hand_free[gangCardInd] = 0
        #             self.history[gangCardInd] = 0
        # elif requests(requestID) == requests.BUGANG:
        #     # 补杠没有在番数计算吗？？？？？
        #     play_card_ind = self.getCardInd(request[-1])
        #     self.history[play_card_ind] += 1
        #     if playerID == myPlayerID:
        #         for id, comb in enumerate(self.hand_fixed_data):
        #             if comb[1] == request[-1]:
        #                 self.hand_fixed_data[id][0] = 'GANG'
        #                 break
        #         self.hand_free[play_card_ind] -= 1
        #         self.hand_fixed[play_card_ind] += 1
        return request

    def maintain_status(self, request, my_free, history, play_history, on_table, last_play, angang):
        requestID = int(request[0])
        playerID = int(request[1])
        player_position = self.player_positions[playerID]
        if requests(requestID) == requests.drawCard:
            my_free[self.getCardInd(request[-1])] += 1
            history[self.getCardInd(request[-1])] += 1
        elif requests(requestID) == requests.PLAY:
            play_card = self.getCardInd(request[-1])
            play_history[player_position][play_card] += 1
            last_play[player_position] = play_card
            # 自己
            if player_position == 0:
                my_free[play_card] -= 1
            else:
                history[play_card] += 1
        elif requests(requestID) == requests.PENG:
            # 上一步一定有play
            last_card_ind = self.getCardInd(self.prev_request[-1])
            play_card_ind = self.getCardInd(request[-1])
            on_table[player_position][last_card_ind] = 3
            play_history[player_position][play_card_ind] += 1
            last_play[player_position] = play_card_ind
            if player_position != 0:
                history[last_card_ind] += 2
                history[play_card_ind] += 1
            else:
                # 记录peng来源于哪个玩家
                last_player = int(self.prev_request[1])
                last_player_pos = self.player_positions[last_player]
                self.hand_fixed_data.append(('PENG', self.prev_request[-1], last_player_pos))
                my_free[last_card_ind] -= 2
                my_free[play_card_ind] -= 1
        elif requests(requestID) == requests.CHI:
            # 上一步一定有play
            last_card_ind = self.getCardInd(self.prev_request[-1])
            middle_card, play_card = request[3:5]
            middle_card_ind = self.getCardInd(middle_card)
            play_card_ind = self.getCardInd(play_card)
            on_table[player_position][middle_card_ind-1:middle_card_ind+2] += 1
            if player_position != 0:
                history[middle_card_ind-1:middle_card_ind+2] += 1
                history[last_card_ind] -= 1
                history[play_card_ind] += 1
            else:
                # CHI,中间牌名，123代表上家的牌是第几张
                self.hand_fixed_data.append(('CHI', middle_card, last_card_ind - middle_card_ind + 2))
                my_free[middle_card_ind-1:middle_card_ind+2] -= 1
                my_free[last_card_ind] += 1
                my_free[play_card_ind] -= 1
        elif requests(requestID) == requests.GANG:
            # 暗杠
            if requests(int(self.prev_request[0])) in [requests.drawCard, requests.DRAW]:
                request[2] = requests.ANGANG.name
                if player_position == 0:
                    gangCard = self.an_gang_card
                    # print(gangCard)
                    if gangCard == '' and not self.botzone:
                        print(self.prev_request)
                        print(request)
                        print(self.fname)
                    gangCardInd = self.getCardInd(gangCard)
                    # 记录gang来源于哪个玩家（可能来自自己，暗杠）
                    self.hand_fixed_data.append(('GANG', gangCard, 0))
                    on_table[0][gangCardInd] = 4
                    my_free[gangCardInd] = 0
                else:
                    angang[player_position] += 1
            else:
                # 明杠
                gangCardInd = self.getCardInd(self.prev_request[-1])
                request[2] = requests.MINGGANG.name
                history[gangCardInd] = 4
                on_table[player_position][gangCardInd] = 4
                if player_position == 0:
                    # 记录gang来源于哪个玩家
                    last_player = int(self.prev_request[1])
                    self.hand_fixed_data.append(
                            ('GANG', self.prev_request[-1], self.player_positions[last_player]))
                    my_free[gangCardInd] = 0
        elif requests(requestID) == requests.BUGANG:
            bugang_card_ind = self.getCardInd(request[-1])
            history[bugang_card_ind] = 4
            on_table[player_position][bugang_card_ind] = 4
            if player_position == 0:
                for id, comb in enumerate(self.hand_fixed_data):
                    if comb[1] == request[-1]:
                        self.hand_fixed_data[id] = ('GANG', comb[1], comb[2])
                        break
                my_free[bugang_card_ind] = 0
        return request

    def build_output(self, response, cards_ind):
        if (responses.need_cards.value[response.value] == 1 and response != responses.CHI) or response == responses.PENG:
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
        if cardName[0] == 'H':
            print('hua ' + self.fname)
        return cards[cardName[0]].value + int(cardName[1]) - 1

    def getCardName(self, cardInd):
        num = 1
        while True:
            if cardInd in cards._value2member_map_:
                break
            num += 1
            cardInd -= 1
        return cards(cardInd).name + str(num)

def train_main():
    # my_bot = MahjongHandler(train=True, model_path='data/naive_model', load_model=False, save_model=False)
    # with open('training_data/Tread 10086-mini.json', 'r') as f:
    #     round_data = json.load(f)[0]
    #     for j in range(4):
    #         train_requests = round_data["requests"][j]
    #         first_request = '0 {} {}'.format(j, round_data['zhuangjia'])
    #         train_requests.insert(0, first_request)
    #         train_responses = ['PASS'] + round_data["responses"][j]
    #         for _request, _response in zip(train_requests, train_responses):
    #             my_bot.step(_request, _response, round_data['fname'])
    #         my_bot.reset()

    my_bot = MahjongHandler(train=True, model_path='model_result/naive_model', load_model=True, save_model=True)
    count = 0
    restore_count = my_bot.round_count
    # restore_count = 2000
    trainning_data_files = os.listdir('training_data')
    for fname in trainning_data_files:
        with open('training_data/{}'.format(fname), 'r') as f:
            rounds_data = json.load(f)
            random.shuffle(rounds_data)
            for round_data in rounds_data:
                for j in range(4):
                    count += 1
                    if count < restore_count:
                        continue
                    if count % 500 == 0:
                        print(count)
                    train_requests = round_data["requests"][j]
                    first_request = '0 {} {}'.format(j, round_data['zhuangjia'])
                    train_requests.insert(0, first_request)
                    train_responses = ['PASS'] + round_data["responses"][j]
                    for _request, _response in zip(train_requests, train_responses):
                        my_bot.step(_request, _response, round_data['fname'])
                        # print(_request, _response, round_data['fname'])
                        # try:
                        #     my_bot.step(_request, _response, round_data['fname'])
                        # except Exception as e:
                        #     print(e)
                        #     print(round_data['fname'])
                            # exit(0)
                    my_bot.reset()
def run_main():
    my_bot = MahjongHandler(train=False, model_path='data/naive_model', load_model=True, save_model=False, botzone=True)
    while True:
        my_bot.step()
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()

if __name__ == '__main__':
    train_main()
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

# round = {"outcome": "\u8352\u5e84", "fanxing": ["\u8352\u5e84-0"], "score": 0, "fname": "C:\\Users\\Administrator\\Desktop\\mjdata\\output2017/LIU/2017-01-08-925.txt", "zhuangjia": 1, "requests": [["1 0 0 0 0 W8 T3 B5 W4 T5 T2 W3 W9 T8 B4 W5 T5 B7", "3 0 DRAW", "3 1 PLAY T1", "3 2 DRAW", "3 2 PLAY T2", "3 3 DRAW", "3 3 PLAY W3", "2 W2", "3 0 PLAY T8", "3 1 DRAW", "3 1 PLAY T2", "3 2 DRAW", "3 2 PLAY W2", "3 3 DRAW", "3 3 PLAY T9", "2 T4", "3 0 PLAY B7", "3 1 DRAW", "3 1 PLAY T3", "3 2 DRAW", "3 2 PLAY J3", "3 3 DRAW", "3 3 PLAY B6", "3 0 CHI B5 W2", "3 1 DRAW", "3 1 PLAY T2", "3 2 DRAW", "3 2 PLAY J2", "3 3 DRAW", "3 3 PLAY F3", "2 W4", "3 0 PLAY W4", "3 1 DRAW", "3 1 PLAY T7", "3 2 DRAW", "3 2 PLAY B4", "3 1 PENG F3", "3 2 DRAW", "3 2 PLAY T1", "3 3 DRAW", "3 3 PLAY W1", "2 B2", "3 0 PLAY B2", "3 3 PENG F1", "2 F3", "3 0 PLAY F3", "3 1 DRAW", "3 1 PLAY W1", "3 2 DRAW", "3 2 PLAY B5", "3 3 DRAW", "3 3 PLAY J2", "2 F2", "3 0 PLAY F2", "3 1 PENG F1", "3 2 DRAW", "3 2 PLAY F2", "3 3 DRAW", "3 3 PLAY T9", "2 W4", "3 0 PLAY W4", "3 1 DRAW", "3 1 PLAY F4", "3 2 DRAW", "3 2 PLAY B2", "3 3 DRAW", "3 3 PLAY W3", "2 W9", "3 0 PLAY W9", "3 1 DRAW", "3 1 PLAY W4", "3 2 DRAW", "3 2 PLAY T1", "3 3 DRAW", "3 3 PLAY J3", "2 F4", "3 0 PLAY F4", "3 1 DRAW", "3 1 PLAY B1", "3 2 DRAW", "3 2 PLAY W2", "3 3 DRAW", "3 3 PLAY F4", "2 B1", "3 0 PLAY B1", "3 1 DRAW", "3 1 PLAY W1", "3 2 DRAW", "3 2 GANG", "3 2 DRAW", "3 2 PLAY W1", "3 3 DRAW", "3 3 PLAY F4", "2 T3", "3 0 PLAY T3", "3 1 DRAW", "3 1 PLAY J3", "3 2 DRAW", "3 2 PLAY F1", "3 3 DRAW", "3 3 PLAY T8", "2 T8", "3 0 PLAY T8", "3 1 DRAW", "3 1 PLAY W5", "3 2 DRAW", "3 2 PLAY W6", "3 3 DRAW", "3 3 PLAY B5", "2 B8", "3 0 PLAY B8", "3 1 DRAW", "3 1 PLAY F3", "3 2 DRAW", "3 2 PLAY T6", "3 3 DRAW", "3 3 PLAY B1", "2 T6", "3 0 PLAY T6", "3 1 DRAW", "3 1 PLAY W9", "3 2 DRAW", "3 2 PLAY B8", "3 3 DRAW", "3 3 PLAY W5", "2 T6", "3 0 PLAY T6", "3 1 DRAW", "3 1 PLAY B1", "3 2 DRAW", "3 2 PLAY T6", "3 3 DRAW", "3 3 PLAY W6", "2 J2", "3 0 PLAY J2", "3 1 DRAW", "3 1 PLAY W6", "3 2 DRAW", "3 2 PLAY W8", "3 3 DRAW", "3 3 PLAY W6", "2 F1", "3 0 PLAY F1", "3 1 DRAW", "3 1 PLAY W3", "3 2 DRAW", "3 2 PLAY T7", "3 3 DRAW", "3 3 PLAY W8", "2 W2", "3 0 PLAY W2", "3 1 DRAW", "3 1 PLAY T5", "3 0 PENG W9", "3 1 DRAW", "3 1 PLAY W5", "3 2 DRAW", "3 2 PLAY J2", "3 3 DRAW", "3 3 PLAY J3", "2 B9", "3 0 PLAY W8", "3 1 DRAW", "3 1 PLAY W7", "3 2 DRAW", "3 2 PLAY T7", "3 3 DRAW", "3 3 PLAY T9", "2 T3", "3 0 PLAY W5", "3 1 DRAW", "3 1 PLAY T1", "3 2 DRAW"], ["1 0 0 0 0 F4 B4 F1 B7 B4 B9 F3 T1 T2 B6 T7 B1 F2", "2 F2", "3 1 PLAY T1", "3 2 DRAW", "3 2 PLAY T2", "3 3 DRAW", "3 3 PLAY W3", "3 0 DRAW", "3 0 PLAY T8", "2 B9", "3 1 PLAY T2", "3 2 DRAW", "3 2 PLAY W2", "3 3 DRAW", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY B7", "2 T3", "3 1 PLAY T3", "3 2 DRAW", "3 2 PLAY J3", "3 3 DRAW", "3 3 PLAY B6", "3 0 CHI B5 W2", "2 T2", "3 1 PLAY T2", "3 2 DRAW", "3 2 PLAY J2", "3 3 DRAW", "3 3 PLAY F3", "3 0 DRAW", "3 0 PLAY W4", "2 B9", "3 1 PLAY T7", "3 2 DRAW", "3 2 PLAY B4", "3 1 PENG F3", "3 2 DRAW", "3 2 PLAY T1", "3 3 DRAW", "3 3 PLAY W1", "3 0 DRAW", "3 0 PLAY B2", "3 3 PENG F1", "3 0 DRAW", "3 0 PLAY F3", "2 W1", "3 1 PLAY W1", "3 2 DRAW", "3 2 PLAY B5", "3 3 DRAW", "3 3 PLAY J2", "3 0 DRAW", "3 0 PLAY F2", "3 1 PENG F1", "3 2 DRAW", "3 2 PLAY F2", "3 3 DRAW", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY W4", "2 B6", "3 1 PLAY F4", "3 2 DRAW", "3 2 PLAY B2", "3 3 DRAW", "3 3 PLAY W3", "3 0 DRAW", "3 0 PLAY W9", "2 W4", "3 1 PLAY W4", "3 2 DRAW", "3 2 PLAY T1", "3 3 DRAW", "3 3 PLAY J3", "3 0 DRAW", "3 0 PLAY F4", "2 B8", "3 1 PLAY B1", "3 2 DRAW", "3 2 PLAY W2", "3 3 DRAW", "3 3 PLAY F4", "3 0 DRAW", "3 0 PLAY B1", "2 W1", "3 1 PLAY W1", "3 2 DRAW", "3 2 GANG", "3 2 DRAW", "3 2 PLAY W1", "3 3 DRAW", "3 3 PLAY F4", "3 0 DRAW", "3 0 PLAY T3", "2 J3", "3 1 PLAY J3", "3 2 DRAW", "3 2 PLAY F1", "3 3 DRAW", "3 3 PLAY T8", "3 0 DRAW", "3 0 PLAY T8", "2 W5", "3 1 PLAY W5", "3 2 DRAW", "3 2 PLAY W6", "3 3 DRAW", "3 3 PLAY B5", "3 0 DRAW", "3 0 PLAY B8", "2 F3", "3 1 PLAY F3", "3 2 DRAW", "3 2 PLAY T6", "3 3 DRAW", "3 3 PLAY B1", "3 0 DRAW", "3 0 PLAY T6", "2 W9", "3 1 PLAY W9", "3 2 DRAW", "3 2 PLAY B8", "3 3 DRAW", "3 3 PLAY W5", "3 0 DRAW", "3 0 PLAY T6", "2 B1", "3 1 PLAY B1", "3 2 DRAW", "3 2 PLAY T6", "3 3 DRAW", "3 3 PLAY W6", "3 0 DRAW", "3 0 PLAY J2", "2 W6", "3 1 PLAY W6", "3 2 DRAW", "3 2 PLAY W8", "3 3 DRAW", "3 3 PLAY W6", "3 0 DRAW", "3 0 PLAY F1", "2 W3", "3 1 PLAY W3", "3 2 DRAW", "3 2 PLAY T7", "3 3 DRAW", "3 3 PLAY W8", "3 0 DRAW", "3 0 PLAY W2", "2 T5", "3 1 PLAY T5", "3 0 PENG W9", "2 W5", "3 1 PLAY W5", "3 2 DRAW", "3 2 PLAY J2", "3 3 DRAW", "3 3 PLAY J3", "3 0 DRAW", "3 0 PLAY W8", "2 W7", "3 1 PLAY W7", "3 2 DRAW", "3 2 PLAY T7", "3 3 DRAW", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY W5", "2 T1", "3 1 PLAY T1", "3 2 DRAW"], ["1 0 0 0 0 W8 J2 J3 B3 J1 T2 T9 T7 B8 B3 W2 W9 B8", "3 1 DRAW", "3 1 PLAY T1", "2 F2", "3 2 PLAY T2", "3 3 DRAW", "3 3 PLAY W3", "3 0 DRAW", "3 0 PLAY T8", "3 1 DRAW", "3 1 PLAY T2", "2 W7", "3 2 PLAY W2", "3 3 DRAW", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY B7", "3 1 DRAW", "3 1 PLAY T3", "2 B3", "3 2 PLAY J3", "3 3 DRAW", "3 3 PLAY B6", "3 0 CHI B5 W2", "3 1 DRAW", "3 1 PLAY T2", "2 J1", "3 2 PLAY J2", "3 3 DRAW", "3 3 PLAY F3", "3 0 DRAW", "3 0 PLAY W4", "3 1 DRAW", "3 1 PLAY T7", "2 B4", "3 2 PLAY B4", "3 1 PENG F3", "2 T1", "3 2 PLAY T1", "3 3 DRAW", "3 3 PLAY W1", "3 0 DRAW", "3 0 PLAY B2", "3 3 PENG F1", "3 0 DRAW", "3 0 PLAY F3", "3 1 DRAW", "3 1 PLAY W1", "2 B5", "3 2 PLAY B5", "3 3 DRAW", "3 3 PLAY J2", "3 0 DRAW", "3 0 PLAY F2", "3 1 PENG F1", "2 T7", "3 2 PLAY F2", "3 3 DRAW", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY W4", "3 1 DRAW", "3 1 PLAY F4", "2 B2", "3 2 PLAY B2", "3 3 DRAW", "3 3 PLAY W3", "3 0 DRAW", "3 0 PLAY W9", "3 1 DRAW", "3 1 PLAY W4", "2 T1", "3 2 PLAY T1", "3 3 DRAW", "3 3 PLAY J3", "3 0 DRAW", "3 0 PLAY F4", "3 1 DRAW", "3 1 PLAY B1", "2 W2", "3 2 PLAY W2", "3 3 DRAW", "3 3 PLAY F4", "3 0 DRAW", "3 0 PLAY B1", "3 1 DRAW", "3 1 PLAY W1", "2 B3", "3 2 GANG", "2 W1", "3 2 PLAY W1", "3 3 DRAW", "3 3 PLAY F4", "3 0 DRAW", "3 0 PLAY T3", "3 1 DRAW", "3 1 PLAY J3", "2 F1", "3 2 PLAY F1", "3 3 DRAW", "3 3 PLAY T8", "3 0 DRAW", "3 0 PLAY T8", "3 1 DRAW", "3 1 PLAY W5", "2 W6", "3 2 PLAY W6", "3 3 DRAW", "3 3 PLAY B5", "3 0 DRAW", "3 0 PLAY B8", "3 1 DRAW", "3 1 PLAY F3", "2 T6", "3 2 PLAY T6", "3 3 DRAW", "3 3 PLAY B1", "3 0 DRAW", "3 0 PLAY T6", "3 1 DRAW", "3 1 PLAY W9", "2 B7", "3 2 PLAY B8", "3 3 DRAW", "3 3 PLAY W5", "3 0 DRAW", "3 0 PLAY T6", "3 1 DRAW", "3 1 PLAY B1", "2 T6", "3 2 PLAY T6", "3 3 DRAW", "3 3 PLAY W6", "3 0 DRAW", "3 0 PLAY J2", "3 1 DRAW", "3 1 PLAY W6", "2 W8", "3 2 PLAY W8", "3 3 DRAW", "3 3 PLAY W6", "3 0 DRAW", "3 0 PLAY F1", "3 1 DRAW", "3 1 PLAY W3", "2 T8", "3 2 PLAY T7", "3 3 DRAW", "3 3 PLAY W8", "3 0 DRAW", "3 0 PLAY W2", "3 1 DRAW", "3 1 PLAY T5", "3 0 PENG W9", "3 1 DRAW", "3 1 PLAY W5", "2 J2", "3 2 PLAY J2", "3 3 DRAW", "3 3 PLAY J3", "3 0 DRAW", "3 0 PLAY W8", "3 1 DRAW", "3 1 PLAY W7", "2 T7", "3 2 PLAY T7", "3 3 DRAW", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY W5", "3 1 DRAW", "3 1 PLAY T1", "2 T4"], ["1 0 0 0 0 W6 W3 F3 T4 F4 B6 T9 F1 F4 J3 B2 W6 W8", "3 1 DRAW", "3 1 PLAY T1", "3 2 DRAW", "3 2 PLAY T2", "2 B2", "3 3 PLAY W3", "3 0 DRAW", "3 0 PLAY T8", "3 1 DRAW", "3 1 PLAY T2", "3 2 DRAW", "3 2 PLAY W2", "2 T4", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY B7", "3 1 DRAW", "3 1 PLAY T3", "3 2 DRAW", "3 2 PLAY J3", "2 W5", "3 3 PLAY B6", "3 0 CHI B5 W2", "3 1 DRAW", "3 1 PLAY T2", "3 2 DRAW", "3 2 PLAY J2", "2 W7", "3 3 PLAY F3", "3 0 DRAW", "3 0 PLAY W4", "3 1 DRAW", "3 1 PLAY T7", "3 2 DRAW", "3 2 PLAY B4", "3 1 PENG F3", "3 2 DRAW", "3 2 PLAY T1", "2 W1", "3 3 PLAY W1", "3 0 DRAW", "3 0 PLAY B2", "3 3 PENG F1", "3 0 DRAW", "3 0 PLAY F3", "3 1 DRAW", "3 1 PLAY W1", "3 2 DRAW", "3 2 PLAY B5", "2 J2", "3 3 PLAY J2", "3 0 DRAW", "3 0 PLAY F2", "3 1 PENG F1", "3 2 DRAW", "3 2 PLAY F2", "2 T9", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY W4", "3 1 DRAW", "3 1 PLAY F4", "3 2 DRAW", "3 2 PLAY B2", "2 W3", "3 3 PLAY W3", "3 0 DRAW", "3 0 PLAY W9", "3 1 DRAW", "3 1 PLAY W4", "3 2 DRAW", "3 2 PLAY T1", "2 W7", "3 3 PLAY J3", "3 0 DRAW", "3 0 PLAY F4", "3 1 DRAW", "3 1 PLAY B1", "3 2 DRAW", "3 2 PLAY W2", "2 J1", "3 3 PLAY F4", "3 0 DRAW", "3 0 PLAY B1", "3 1 DRAW", "3 1 PLAY W1", "3 2 DRAW", "3 2 GANG", "3 2 DRAW", "3 2 PLAY W1", "2 T8", "3 3 PLAY F4", "3 0 DRAW", "3 0 PLAY T3", "3 1 DRAW", "3 1 PLAY J3", "3 2 DRAW", "3 2 PLAY F1", "2 J1", "3 3 PLAY T8", "3 0 DRAW", "3 0 PLAY T8", "3 1 DRAW", "3 1 PLAY W5", "3 2 DRAW", "3 2 PLAY W6", "2 B5", "3 3 PLAY B5", "3 0 DRAW", "3 0 PLAY B8", "3 1 DRAW", "3 1 PLAY F3", "3 2 DRAW", "3 2 PLAY T6", "2 B1", "3 3 PLAY B1", "3 0 DRAW", "3 0 PLAY T6", "3 1 DRAW", "3 1 PLAY W9", "3 2 DRAW", "3 2 PLAY B8", "2 B7", "3 3 PLAY W5", "3 0 DRAW", "3 0 PLAY T6", "3 1 DRAW", "3 1 PLAY B1", "3 2 DRAW", "3 2 PLAY T6", "2 T5", "3 3 PLAY W6", "3 0 DRAW", "3 0 PLAY J2", "3 1 DRAW", "3 1 PLAY W6", "3 2 DRAW", "3 2 PLAY W8", "2 B6", "3 3 PLAY W6", "3 0 DRAW", "3 0 PLAY F1", "3 1 DRAW", "3 1 PLAY W3", "3 2 DRAW", "3 2 PLAY T7", "2 J3", "3 3 PLAY W8", "3 0 DRAW", "3 0 PLAY W2", "3 1 DRAW", "3 1 PLAY T5", "3 0 PENG W9", "3 1 DRAW", "3 1 PLAY W5", "3 2 DRAW", "3 2 PLAY J2", "2 B5", "3 3 PLAY J3", "3 0 DRAW", "3 0 PLAY W8", "3 1 DRAW", "3 1 PLAY W7", "3 2 DRAW", "3 2 PLAY T7", "2 T9", "3 3 PLAY T9", "3 0 DRAW", "3 0 PLAY W5", "3 1 DRAW", "3 1 PLAY T1", "3 2 DRAW"]], "responses": [["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B7", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "CHI B5 W2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B2", "PASS", "PASS", "PLAY F3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W9", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W2", "PASS", "PASS", "PENG W9", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W5", "PASS", "PASS", "PASS", "PASS"], ["PASS", "PLAY T1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T7", "PASS", "PASS", "PENG F3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PENG F1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W5", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W9", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T5", "PASS", "PASS", "PLAY W5", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W7", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T1", "PASS", "PASS"], ["PASS", "PASS", "PASS", "PLAY T2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B4", "PASS", "PASS", "PLAY T1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B5", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "GANG B3", "PASS", "PLAY W1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T7", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T7", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T4"], ["PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T9", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W1", "PASS", "PASS", "PENG F1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J2", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T9", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J3", "PASS", "PASS", "HU", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY F4", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B5", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY B1", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W5", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W6", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY W8", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY J3", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PLAY T9", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"]]}
# for j in range(4):
#     count += 1
#     # if count < 2000:
#     #     continue
#     if count % 500 == 0:
#         print(count)
#     train_requests = round["requests"][j]
#     first_request = '0 {} {}'.format(j, round['zhuangjia'])
#     train_requests.insert(0, first_request)
#     train_responses = ['PASS'] + round["responses"][j]
#     for _request, _response in zip(train_requests, train_responses):
#         print(_request, _response)
#         try:
#             my_bot.step(_request, _response, round['fname'])
#         except Exception as e:
#             print(e)
#             print(round['fname'])
#             # exit(0)
#     my_bot.reset()

# inputs = {
# 	"requests": [
# 		"0 3 3",
# 		"1 0 0 0 0 W1 B1 T3 W4 B1 T8 F1 J1 W5 B3 W8 F3 W9",
# 		"3 0 DRAW",
# 		"3 0 PLAY F2",
# 		"3 1 DRAW",
# 		"3 1 PLAY F1",
# 		"3 2 DRAW",
# 		"3 2 PLAY F4",
# 		"2 T1",
# 		"3 3 PLAY F1",
# 		"3 0 DRAW",
# 		"3 0 PLAY J1",
# 		"3 1 DRAW",
# 		"3 1 PLAY F4",
# 		"3 2 DRAW",
# 		"3 2 PLAY W8",
# 		"2 F4",
# 		"3 3 PLAY F4",
# 		"3 0 DRAW",
# 		"3 0 PLAY F3",
# 		"3 1 DRAW",
# 		"3 1 PLAY J3",
# 		"3 2 DRAW",
# 		"3 2 PLAY W9",
# 		"2 T9",
# 		"3 3 PLAY F3",
# 		"3 0 DRAW",
# 		"3 0 PLAY W9",
# 		"3 1 DRAW",
# 		"3 1 PLAY J3",
# 		"3 2 DRAW",
# 		"3 2 PLAY W6",
# 		"3 3 CHI W5 J1",
# 		"3 0 DRAW",
# 		"3 0 PLAY F2",
# 		"3 1 DRAW",
# 		"3 1 PLAY W9",
# 		"3 2 DRAW",
# 		"3 2 PLAY F4",
# 		"2 T2",
# 		"3 3 PLAY W9",
# 		"3 0 DRAW",
# 		"3 0 PLAY T3",
# 		"3 1 DRAW",
# 		"3 1 PLAY B8",
# 		"3 2 DRAW",
# 		"3 2 PLAY T7",
# 		"3 3 CHI T8 B1",
# 		"3 0 DRAW",
# 		"3 0 PLAY B3",
# 		"3 1 DRAW",
# 		"3 1 PLAY W2",
# 		"3 2 DRAW",
# 		"3 2 PLAY B1",
# 		"2 W2",
# 		"3 3 PLAY W8",
# 		"3 0 DRAW",
# 		"3 0 PLAY B4",
# 		"3 1 DRAW",
# 		"3 1 PLAY B5",
# 		"3 2 DRAW",
# 		"3 2 PLAY B6",
# 		"2 B6",
# 		"3 3 PLAY W2",
# 		"3 0 CHI W2 W4",
# 		"3 1 DRAW",
# 		"3 1 PLAY F3",
# 		"3 2 DRAW",
# 		"3 2 PLAY W4",
# 		"2 T4",
# 		"3 3 PLAY W1",
# 		"3 0 DRAW",
# 		"3 0 PLAY J2",
# 		"3 1 DRAW",
# 		"3 1 PLAY B5",
# 		"3 2 DRAW",
# 		"3 2 PLAY W8",
# 		"2 W4",
# 		"3 3 PLAY W4",
# 		"3 0 DRAW",
# 		"3 0 PLAY B5",
# 		"3 1 DRAW",
# 		"3 1 PLAY W1",
# 		"3 2 DRAW",
# 		"3 2 PLAY T9"
# 	],
# 	"responses": [
# 		"PASS",
# 		"PASS",
# 		"PASS",
# 		"PASS"
# 	]
# }