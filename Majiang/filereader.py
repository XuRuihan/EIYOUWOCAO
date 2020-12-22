#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software:XXXX
@file: filereader.py
@time: 2020/12/19 18:34
@desc:
'''
# _*_coding:utf-8_*_
import time, threading
import os
import json
'''
Reader类，继承threading.Thread
@__init__方法初始化
@run方法实现了读文件的操作
'''

def round(outcome, fanxing, score, fname, zhuangjia, requests, responses):
    return {
    'outcome' : outcome,
    'fanxing' : fanxing,
    'score' : score,
    'fname' : fname,
    'zhuangjia': zhuangjia,
    'requests' : requests,
    'responses' : responses
    }

class Reader(threading.Thread):
    def __init__(self, files, id):
        super(Reader, self).__init__()
        self.files = files
        self.id = id

    def run(self):
        self.rounds = []
        self.removed = 0
        for filenum, file in enumerate(self.files):
            if filenum % 100 == 0:
                print('{}/{}'.format(filenum, len(self.files)))
            with open(file, encoding='utf-8') as f:
                line = f.readline()
                count = 1
                requests = [[], [], [], []]
                responses = [[], [], [], []]
                zhuangjia = 0
                outcome = ''
                fanxing = []
                score = 0
                fname = file
                flag = True
                while line:
                    # print(line)
                    line = line.strip('\n').split('\t')
                    if count == 2:
                        # print(line)
                        outcome = line[-1]
                        fanxing = list(map(lambda x: x.strip("'"), line[2].strip('[]').split(',')))
                        score = int(line[1])
                        # quan = line[0]
                        # print(fanxing)
                    if count > 2 and count <= 6:
                        playerID = int(line[0])
                        request = "1 0 0 0 0 "
                        cards = line[1]
                        cards = list(map(lambda x: x.strip("'"), cards.strip('[]').split(',')))
                        if len(cards) == 14:
                            draw_card = cards.pop()
                            for card in cards:
                                if 'H' in card:
                                    draw_card = card
                            zhuangjia = playerID
                        else:
                            draw_card = None
                        request += (' '.join(cards))
                        requests[playerID].append(request)
                        if draw_card is not None:
                           requests[playerID].append('2 ' + draw_card)
                           responses[playerID].append("PASS")
                        else:
                            requests[playerID].append('3 {} DRAW'.format(zhuangjia))
                            responses[playerID].append("PASS")
                    if count > 6:
                        playerID = int(line[0])
                        action = line[1]
                        cards = line[2]
                        if action == '吃':
                            middel_card = list(map(lambda x: x.strip("'"), cards.strip('[]').split(',')))[1]
                            next_line = f.readline()
                            play_card = next_line.strip('\n').split('\t')[2]
                            play_card = play_card.strip("[]'")
                            _cards = [middel_card, play_card]
                            _action = action
                        elif action == '碰':
                            next_line = f.readline()
                            play_card = next_line.strip('\n').split('\t')[2]
                            play_card = play_card.strip("[]'")
                            _cards = [play_card]
                            _action = action
                        elif action == '补花':
                            flag = False
                            for i in range(4):
                                if 'H' in requests[i].pop():
                                    flag = True
                                if len(responses[i]) == 0:
                                    print(fname)
                                responses[i].pop()
                            if not flag:
                                self.removed += 1
                                break
                            line = f.readline()
                            if not line:
                                for i in range(4):
                                    requests[i].pop()
                                last_hua = False
                                for i in range(4):
                                    if 'H' in requests[i][-1]:
                                        last_hua = True
                                if last_hua:
                                    for i in range(4):
                                        requests[i].pop()
                                        responses[i].pop()
                            count += 1
                            continue
                        else:
                            card = list(map(lambda x: x.strip("'"), cards.strip('[]').split(',')))[0]
                            _cards = [card]
                            _action = action
                        for i in range(4):
                            request = get_request(_action, playerID, _cards, i)
                            # print(request)
                            response = get_response(_action, playerID, _cards, i)
                            # print(response)
                            if request is not None:
                                # print(request)
                                requests[i].append(request)
                            if response is not None:
                                responses[i].append(response)

                    line = f.readline()
                    if not line:
                        for i in range(4):
                            requests[i].pop()
                        last_hua = False
                        for i in range(4):
                            if 'H' in requests[i][-1]:
                                last_hua = True
                        if last_hua:
                            for i in range(4):
                                requests[i].pop()
                                responses[i].pop()
                    count += 1
                if flag:
                    self.rounds.append(round(outcome, fanxing, score, fname, zhuangjia, requests, responses))

    def get_res(self):
        with open('Tread {}-mini.json'.format(self.id), 'w') as file_obj:
            json.dump(self.rounds, file_obj)
        return self.removed

def get_request(action, playerid, cards, myplayerid):
    playerid = str(playerid)
    myplayerid = str(myplayerid)
    request = None
    if action == '打牌':
        request = ['3', playerid, 'PLAY', cards[0]]
    if action == '摸牌' or action == '补花后摸牌' or action == '杠后摸牌':
        if playerid == myplayerid:
            request = ['2', cards[0]]
        else:
            request = ['3', playerid, 'DRAW']
    if action == '吃':
        request = ['3', playerid, 'CHI'] + cards
    if action == '碰':
        request = ['3', playerid, 'PENG', cards[0]]
    if action == '明杠' or action == '暗杠':
        request = ['3', playerid, 'GANG']
    if action == '补杠':
        request = ['3', playerid, 'BUGANG', cards[0]]
    if request is None:
        return None
    return ' '.join(request)

def get_response(action, playerid, cards, myplayerid):
    if playerid != myplayerid:
        response = ['PASS']
    else:
        if action == '打牌':
            response = ['PLAY', cards[0]]
        if action == '摸牌' or action == '补花后摸牌' or action == '杠后摸牌':
           response = ['PASS']
        if action == '吃':
            response = ['CHI'] + cards
        if action == '碰':
            response = ['PENG', cards[0]]
        if action == '明杠':
            response = ['GANG']
        if action == '暗杠':
            response = ['GANG', cards[0]]
        if action == '补杠':
            response = ['BUGANG', cards[0]]
        if action == '和牌':
            response = ['HU']
    # print(action)
    return ' '.join(response)


if __name__ == '__main__':
    #线程数量
    thread_num = 4
    #起始时间
    t = []
    folder = 'C:\\Users\\Administrator\\Desktop\\mjdata\\output2017'
    dirs = os.listdir(folder)
    files = []
    for dir in dirs:
        subfolder = folder + '/' + dir
        for file in os.listdir(subfolder):
            files.append(subfolder + '/' + file)
    files = files[:10000]
    filenum = len(files) // thread_num
    #生成线程
    for i in range(thread_num):
        t.append(Reader(files[filenum*i:filenum*(i+1)], i))
    #开启线程
    for i in range(thread_num):
        t[i].start()
    sum_rm = 0
    for i in range(thread_num):
        t[i].join()
        sum_rm += t[i].get_res()
    print(sum_rm)
    #结束时间