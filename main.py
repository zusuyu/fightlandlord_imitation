import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from collections import OrderedDict










def getCardId(card):
    # 求一张牌的 id
    if card < 52:
        return card // 4
    else:
        return card - 39

def getCombo(cards):
    # a combo is represented as a tuple(k, l, r, w)
    # 表示有 k * [l, r] 即 k 张 [l, r] 中的牌（作为主体）w \in [0, 1, 2] 表示带的是啥类型
    if len(cards) == 0:
        return (0, 0, 0, 0)
    tmp = np.zeros(15, dtype = int)
    for card in cards:
        tmp[getCardId(card)] += 1
    k = np.max(tmp)
    l = np.min(np.where(tmp == k))
    r = np.max(np.where(tmp == k))
    w = 0
    if k == 3:
        w = len(cards) // (r - l + 1) - 3
    if k == 4:
        w = (len(cards) // (r - l + 1) - 4) // 2
    return (k, l, r, w)

combo_dict = {}
combo_list = []
combo_cnt = 0

def initCombo():
    global combo_dict, combo_list, combo_cnt
    combo_dict = {}
    combo_list = []
    combo_cnt = 0
    def addCombo(combo):
        global combo_dict, combo_list, combo_cnt
        combo_list.append(combo)
        combo_dict[combo] = combo_cnt
        combo_cnt += 1

    minLength = [0, 5, 3, 2, 2]
    maxWings = [0, 1, 1, 3, 3]
    fold = [0, 0, 0, 1, 2]
    for k in range(1, 5):
        for x in range(13):
            for w in range(maxWings[k]):
                addCombo((k, x, x, w))
        for l in range(12):
            for r in range(l + minLength[k] - 1, 12):
                for w in range(maxWings[k]):
                    if (r - l + 1) * (k + w * fold[k]) <= 20:
                        addCombo((k, l, r, w))
    addCombo((1, 13, 13, 0))
    addCombo((1, 14, 14, 0))
    addCombo((1, 13, 14, 0))
    addCombo((0, 0, 0, 0))
    
initCombo()

def getPartition(cards):
    # 把一次出牌的编号集合划分成 mainbody 和 bywings
    # 其中 mainbody 是一个 list ，bywings 中每个 wing 是一个 list ，也就是一个 list 的 list
    combo = getCombo(cards)
    tmp = [[] for i in range(15)]
    for card in cards:
        tmp[getCardId(card)].append(card)
    mainbody, bywings = [], []
    for i in range(15):
        if len(tmp[i]) > 0:
            if combo[1] <= i and i <= combo[2]:
                mainbody.extend(tmp[i])
            else:
                bywings.append(tmp[i])
    return mainbody, bywings

def getComboMask(combo):
    # 给出一个 combo ，返回可以接在其后面牌型 mask 
    mask = np.zeros(combo_cnt)
    if combo == (0, 0, 0, 0):
        mask = np.ones(combo_cnt)
        mask[combo_dict[(0, 0, 0, 0)]] = 0
        return mask
    mask[combo_dict[(0, 0, 0, 0)]] = 1

    if combo == (1, 13, 14, 0):
        return mask
    mask[combo_dict[(1, 13, 14, 0)]] = 1

    if combo[0] == 4 and combo[1] == combo[2] and combo[3] == 0:
        for i in range(combo[1] + 1, 13):
            mask[combo_dict[(4, i, i, 0)]] = 1
        return mask
    for i in range(13):
        mask[combo_dict[(4, i, i, 0)]] = 1

    for cb in combo_list:
        if cb[0] == combo[0] and cb[2] - cb[1] == combo[2] - combo[1] and cb[3] == combo[3] and cb[1] > combo[1]:
            mask[combo_dict[cb]] = 1
            
    return mask










class Game(object):
    # 这里 0 始终是地主，1 始终是地主下家，2 始终是地主上家

    def __init__(self, init_data):
        self.hand = [np.zeros(15, dtype = int), np.zeros(15, dtype = int), np.zeros(15, dtype = int)]
        for player in range(3):
            for card in init_data[player]:
                self.hand[player][getCardId(card)] += 1
    
    def play(self, player, cards):
        # 模拟打牌 打出 cards 这个 list 中的所有牌
        for card in cards:
            self.hand[player][getCardId(card)] -= 1
            
    def possess(self, player, combo):
        # 判断 player 这个玩家是否拥有 combo 这个牌型的牌
        if combo == (0, 0, 0, 0):
            return True
        for i in range(combo[1], combo[2] + 1):
            if self.hand[player][i] < combo[0]:
                return False
            
        fold = [0, 0, 0, 1, 2]
        need_wings = (combo[2] - combo[1] + 1) * fold[combo[0]] if combo[3] > 0 else 0
        for i in range(15):
            if i < combo[1] or i > combo[2]:
                if self.hand[player][i] >= combo[3]:
                    need_wings -= 1
        if need_wings > 0:
            return False
        return True
    
    def getPossessMask(self, player):
        # 返回 player 拥有的牌型 mask
        mask = np.zeros(combo_cnt)
        for i in range(combo_cnt):
            if self.possess(player, combo_list[i]) == True:
                mask[i] = 1
        return mask
    
    def getMask1(self, player, combo):
        # getPossessMask 和 getComboMask 取交集
        return self.getPossessMask(player) * getComboMask(combo)
    
    def getMask2(self, player, combo, already_played):
        # 带翼的 mask，哪些翼是可以打的？
        # mask 的大小是 15 或者 13, 表示 15 种单牌和 13 种对子
        # 指明 combo 后：(1)少于1/2张的不能打 (2)和主体部分重复的不能打 (3)打过的不能打
        mask = np.ones(15 if combo[3] == 1 else 13)
        for i in range(mask.shape[0]):
                if self.hand[player][i] < combo[3]:
                    mask[i] = 0
        mask[range(combo[1], combo[2] + 1)] = 0
        mask[already_played] = 0
        return mask
    
    def getInput(self, player):
        # 返回两个网络的输入
        # 这里包含五个部分：我自己每种牌的数量、对手每种牌的数量、我的顺子情况、三个人还剩多少张牌、我拥有牌型的 mask
        # size = 4 * 15 + 4 * 15 + 4 * 12 + 3 * 20 + 379
        p1 = (player + 1) % 3
        p2 = (player + 2) % 3

        myhand = np.zeros((4, 15))
        othershand = np.zeros((4, 15))
        for i in range(4):
            myhand[i, np.where(self.hand[player] == i + 1)] = 1
            othershand[i, np.where(self.hand[p1] + self.hand[p2] == i + 1)] = 1
        
        mystraight = np.zeros((4, 12))
        for i in range(4):
            k = 0
            for j in range(12):
                if self.hand[player][i] >= i + 1:
                    k += 1
                else:
                    k = 0
                mystraight[i, j] = k
                
        handcnt = np.zeros((3, 20))
        for player in range(3):
            handcnt[player, np.sum(self.hand[player]) - 1] = 1

        return np.concatenate([myhand.flatten(), othershand.flatten(), mystraight.flatten(), handcnt.flatten(), self.getPossessMask(player)])
    
_input_size = 60 + 60 + 48 + 60 + combo_cnt









HIDDEN_SIZE = 512
torch.set_default_tensor_type(torch.DoubleTensor)

class MyModule(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(MyModule, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(INPUT_SIZE, HIDDEN_SIZE)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(p = 0.2)),
                ('fc2', nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE))
            ]))
        
    def forward(self, x, m):
        return nn.LogSoftmax(dim = -1)(self.fc(x) * m)
    
    
    
    
    
    
    
    
    
    
_BOTZONE_ONLINE = os.environ.get("USER", "") == "root"

my_hand = []
g = Game([[], [], []])
my_pos = -1
others = []
las_combo = (0, 0, 0, 0)

def BIDDING():
    bid_val = 0
    print(json.dumps({
        "response": bid_val
    }))
    if not _BOTZONE_ONLINE:
        assert(0)
    exit()

model_path = "./data/fightlandlord_model/" if _BOTZONE_ONLINE else "./model/"

def PLAYING():
    def getFromHand(idx):
        global my_hand
        for c in my_hand:
            if getCardId(c) == idx:
                my_hand.remove(c)
                return c
    
    to_play = []
    
    model_name = "best_model_for_" + str(my_pos) + "mainbody.pt"
    model = MyModule(_input_size, combo_cnt)
    model.load_state_dict(torch.load(model_path + model_name))
    
    mask = g.getMask1(my_pos, las_combo)
    combo_id = -1
    if np.sum(mask) == 1:
        combo_id = np.argmax(mask)
    else:
        combo_id = model(torch.from_numpy(g.getInput(my_pos)).unsqueeze(0),
                         torch.from_numpy(mask)).unsqueeze(0).detach().numpy().argmax()
    
    combo = combo_list[combo_id]
    for i in range(combo[1], combo[2] + 1):
        for j in range(combo[0]):
            to_play.append(getFromHand(i))
    g.play(my_pos, to_play)
    
    if combo[3] != 0:
        model_name = "best_model_for_" + str(combo[3] - 1) + "bywings.pt"
        model = MyModule(_input_size, (15 if combo[3] == 1 else 13))
        model.load_state_dict(torch.load(model_path + model_name))

        cnt = (combo[2] - combo[1] + 1) * (1 if combo[0] == 3 else 2)
        already_played = []
        for i in range(cnt):
            wing_id = model(torch.from_numpy(g.getInput(my_pos)),
                            torch.from_numpy(g.getMask2(my_pos, combo, already_played))).detach().numpy().argmax()
            tmp = []
            if wing_id < 15:
                tmp = [getFromHand(wing_id)]
            else:
                wing_id -= 15
                tmp = [getFromHand(wing_id), getFromHand(wing_id)]
            g.play(my_pos, tmp)
            to_play.extend(tmp)
            already_played.append(wing_id)
    
    print(json.dumps({
        "response": to_play
    }))
    if not _BOTZONE_ONLINE:
        assert 0
    exit()
    
if __name__ == "__main__":
    initCombo()
    data = json.loads(input())
    my_hand, others_hand = data["requests"][0]["own"], []
    for i in range(54):
        if i not in my_hand:
            others_hand.append(i)

    TODO = "bidding"
    if "bid" in data["requests"][0]:
        bid_list = data["requests"][0]["bid"]
    
    for i in range(len(data["requests"])):
        request = data["requests"][i]
        
        if "publiccard" in request:
            bot_pos = request["pos"]
            lord_pos = request["landlord"]
            my_pos = (bot_pos - lord_pos + 3) % 3
            others = [(my_pos + 1) % 3, (my_pos + 2) % 3]
            tmp = [[], [], []]
            if my_pos == 0:
                my_hand.extend(request["publiccard"])
                tmp[0] = my_hand
                tmp[1], tmp[2] = others_hand[:17], others_hand[17:] # 随便分
            else:
                tmp[my_pos] = my_hand
                tmp[0] = others_hand[:20]
                tmp[2 if my_pos == 1 else 1] = others_hand[20:]
            g = Game(tmp)
            
        if "history" in request:
            history = request["history"]
            TODO = "playing"
            for j in range(2):
                p = others[j]
                cards = history[j]
                g.play(p, cards)
                cur_combo = getCombo(cards)
                if cur_combo != (0, 0, 0, 0):
                    las_combo = cur_combo

            if i < len(data["requests"]) - 1:
                cards = data["responses"][i]
                g.play(my_pos, cards)
                for c in cards:
                    my_hand.remove(c)
                las_combo = (0, 0, 0, 0)
        
    if TODO == "bidding":
        BIDDING()
    else:
        PLAYING()
