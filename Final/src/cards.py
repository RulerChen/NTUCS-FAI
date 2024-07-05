from itertools import combinations
from collections import Counter

def card_value(value):
    if value.isdigit():
        return int(value)
    else:
        return {'J': 11, 'Q': 12, 'K': 13, 'A': 14}[value]

def is_straight(values):
    values = sorted(values)
    if values == [2, 3, 4, 5, 14]:  
        return True
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))

def is_straight_flush(cards):
    suits = [card[0] for card in cards]
    values = sorted([card_value(card[1]) for card in cards])
    return len(set(suits)) == 1 and is_straight(values)

def is_four_of_a_kind(values):
    counter = Counter(values)
    return 4 in counter.values()

def is_full_house(values):
    counter = Counter(values)
    return 3 in counter.values() and 2 in counter.values()

def is_flush(cards):
    suits = [card[0] for card in cards]
    return len(set(suits)) == 1

def is_straight_cards(cards):
    values = sorted([card_value(card[1]) for card in cards])
    return is_straight(values)

def is_three_of_a_kind(values):
    counter = Counter(values)
    return 3 in counter.values()

def is_two_pair(values):
    counter = Counter(values)
    return len([v for v in counter.values() if v == 2]) == 2

def is_one_pair(values):
    counter = Counter(values)
    return 2 in counter.values()

def evaluate_hand(cards):
    values = [card_value(card[1]) for card in cards]

    if is_straight_flush(cards):
        return 9, max(values)
    elif is_four_of_a_kind(values):
        return 8, [v for v, count in Counter(values).items() if count == 4][0]
    elif is_full_house(values): # 葫蘆
        return 7, [v for v, count in Counter(values).items() if count == 3][0]
    elif is_flush(cards): # 同花
        return 6, max(values)
    elif is_straight_cards(cards):
        return 5, max(values)
    elif is_three_of_a_kind(values):
        return 4, [v for v, count in Counter(values).items() if count == 3][0]
    elif is_two_pair(values):
        return 3, max([v for v, count in Counter(values).items() if count == 2])
    elif is_one_pair(values):
        return 2, [v for v, count in Counter(values).items() if count == 2][0]
    else:
        return 1, max(values)

def get_best_hand(cards):
    # hand cards
    values = [card[1] for card in cards]
    if len(cards) == 2:
        if card_value(cards[0][1]) == card_value(cards[1][1]):
            return 2, card_value(cards[0][1])
        else:
            return 1, max([card_value(cards[0][1]), card_value(cards[1][1])])
    elif len(cards) == 3:
        if is_three_of_a_kind(values):
            return 4, [v for v, count in Counter(values).items() if count == 3][0]
        elif is_one_pair(values):
            return 2, [v for v, count in Counter(values).items() if count == 2][0]
        else:
            return 1, max([card_value(card[1]) for card in cards])
    elif len(cards) == 4:
        if is_three_of_a_kind(values):
            return 4, [v for v, count in Counter(values).items() if count == 3][0]
        elif is_two_pair(values):
            return 3, max([v for v, count in Counter(values).items() if count == 2])
        elif is_one_pair(values):
            return 2, [v for v, count in Counter(values).items() if count == 2][0]
        else:
            return 1, max([card_value(card[1]) for card in cards])



    best_rank = 1
    best_num = 0
    for combination in combinations(cards, 5):
        rank, num = evaluate_hand(combination)
        if rank > best_rank:
            best_rank = rank
            best_num = num
        elif rank == best_rank:
            if num > best_num:
                best_num = num
    return best_rank, best_num

# 最多同花的數量
def most_flush(cards):
    suits = [card[0] for card in cards]
    return max([suits.count(suit) for suit in suits])

# 最長的順子的長度
def longest_straight(cards):
    values = sorted([card_value(card[1]) for card in cards])
    values = list(set(values))
    values.sort()

    longest = 1
    length = 1
    gaps = 0

    for i in range(1, len(values)):
        if values[i] == values[i - 1] + 1:
            length += 1
        elif values[i] > values[i - 1] + 1:
            gaps += values[i] - values[i - 1] - 1
            length = 1

        if length + gaps > longest:
            longest = length + gaps

    return longest

# spade 黑桃
# heart 紅心
# diamond 方塊
# club 梅花

