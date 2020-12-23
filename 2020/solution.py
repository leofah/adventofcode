#!/usr/bin/env python3

import sys
import re
import functools
import operator
import itertools
import numpy as np
import functools
from pprint import pprint
from copy import deepcopy
from collections import defaultdict

def day1():
    numbers = list(map(lambda x: int(x), open('1.txt').read().strip().split('\n')))
    print(list(filter(lambda x: x[1] == 2020, [(a * b, a + b) for a in numbers for b in numbers]))[0][0])
    print(list(filter(lambda x: x[1] == 2020, [(a * b * c, a + b + c) for a in numbers for b in numbers for c in numbers]))[0][0])

def day2():
    matches = [re.search('^(\d*)-(\d*) (.): (\w*)$', x).groups() for x in open('2.txt').read().strip().split('\n')]
    print(len(list(filter(lambda x: x[3].count(x[2]) >= int(x[0]) and x[3].count(x[2]) <= int(x[1]), matches))))
    print(len(list(filter(lambda x: (x[3][int(x[0]) - 1] == x[2]) != (x[3][int(x[1]) - 1] == x[2]) , matches))))

def day3(right = 3, down = 1):
    return len(list(filter(lambda x: x[1][(x[0]*(right)) % len(x[1])] == '#', enumerate(open('3.txt').read().strip().split('\n')[::down]))))

def day4():
    ids = ['byr','iyr','eyr', 'hgt', 'hcl', 'ecl', 'pid']
    passports = open('4.txt').read().strip().split('\n\n')
    print(len(list(filter(lambda x: [i in x for i in ids] == [True for i in ids], passports))))

    def corPassport(passport):
        try:
            byr = int(re.search('byr:(\d*)', passport).groups()[0])
            iyr = int(re.search('iyr:(\d*)', passport).groups()[0])
            eyr = int(re.search('eyr:(\d*)', passport).groups()[0])
            try:
                hgtcm = int(re.search('hgt:(\d*)cm', passport).groups()[0])
            except:
                hgtcm = 0
            try:
                hgtin = int(re.search('hgt:(\d*)in', passport).groups()[0])
            except:
                hgtin = 0
            hcl = re.search('hcl:#([0-9a-f]*)', passport).groups()[0]
            ecl = re.search('ecl:(amb|blu|brn|gry|grn|hzl|oth)', passport).groups()[0]
            pid = re.search('pid:([0-9]*)', passport).groups()[0]
            
            correct = True
            correct &= 1920 <= byr <= 2002
            correct &= 2010 <= iyr <= 2020
            correct &= 2020 <= eyr <= 2030
            correct &= 150 <= hgtcm <= 193 or 59 <= hgtin <= 76
            correct &= len(hcl) == 6
            correct &= len(pid) == 9
            return correct
        except:
            #print(False)
            return False

    print(len(list(filter(lambda x: corPassport(x), passports))))

def day5():
    seats = list(map(lambda x: int(x.replace('R', '1').replace('L', '0').replace('B', '1').replace('F', '0') , 2), open('5.txt').read().strip().split('\n')))
    print(max(seats))
    print(list(filter(lambda x: x + 1 not in seats and x + 2 in seats, seats))[0] + 1)

def day6():
    groups = open('6.txt').read().strip().split('\n\n')
    print(sum([len({a for a in x.replace('\n','')}) for x in groups]))
    print(sum(list(len(functools.reduce(lambda x, y: x.intersection(y), [{a for a in line} for line in x.split('\n')], set(map(chr, range(97,123))))) for x in groups)))
    
def day7():
    rules_input = open('7.txt').read().strip().split('\n')
    rules = {}
    for rule in rules_input:
        subrules = rule.split(',')
        words = subrules[0].split(' ')
        color = words[0] + ' ' + words[1]
        dic = {}
        if words[-3] == 'no':
            rules[color] = dic
            continue
        for subrule in subrules:
            words = subrule.split(' ')
            nrContain = int(words[-4])
            incolor = words[-3] + ' ' + words[-2]
            dic[incolor] = nrContain
        rules[color] = dic
    
    # part 1
    checkedColors = []
    workQueue = ['shiny gold']

    while len(workQueue) > 0:
        work = workQueue.pop()
        if work in checkedColors:
            continue
        checkedColors.append(work)
        for rule in rules:
            if work in rules[rule]:
                workQueue.append(rule)
    
    print(len(checkedColors) - 1)

    # part 2
    contains = {}
    def getNrContainedBags(color):
        if color in contains:
            return contains[color]
        rule = rules[color]
        res = 0
        for bags in rule:
            res += rule[bags] + rule[bags] * getNrContainedBags(bags) 
        contains[color] = res
        return res

    print(getNrContainedBags('shiny gold'))

def day8():
    instructions = open('8.txt').read().strip().split('\n')
    def runInstruction(acc, pc, seen): return acc if pc in seen else (runInstruction(acc + (int(instructions[pc][3:]) if instructions[pc][:3] == 'acc' else 0), pc + (int(instructions[pc][3:]) if instructions[pc][:3] == 'jmp' else 1), seen + [pc]))
    print(runInstruction(0, 0, []))
    
    sol = []
    for i in range(len(instructions)):
        def runInstruction(acc, pc, seen): return [False, acc] if pc in seen else [True, acc] if pc == len(instructions) else (runInstruction(acc + (int(instructions[pc][3:]) if instructions[pc][:3] == 'acc' else 0), pc + (int(instructions[pc][3:]) if (instructions[pc][:3] == 'jmp' and i != pc or instructions[pc][:3] =='nop' and i == pc) else 1), seen + [pc]))
        sol.append(runInstruction(0, 0, []))
    print(list(filter(lambda x: x[0], sol))[0][1])

def day9():
    numbers = [int(x) for x in open('9.txt').read().strip().split('\n')]
    incorrect = list(filter(lambda x: x[1] not in [a + b[1] for b in enumerate(numbers[x[0]:x[0] + 25]) for a in numbers[x[0] + b[0]: x[0] + 25]], enumerate(numbers[25:])))[0][1]
    print(incorrect)

    # this is in O(n^3). But the range is found in one line of code.
    rang = list(filter(lambda x: sum(numbers[x[0]:x[1]]) == incorrect, [[a,b] for a in range(len(numbers)) for b in range(a, len(numbers))]))[0]
    print(min(numbers[rang[0]:rang[1]]) + max(numbers[rang[0]:rang[1]]))

    # this is in linear time and solves the problem as well
    summ = 0
    start, end = 0, 0
    while summ != incorrect:
        if summ < incorrect:
            summ += numbers[end]
            end += 1
        else:
            summ -= numbers[start]
            start += 1

    print(min(numbers[start:end]) + max(numbers[start:end]))

def day10():
    adapters = [0] + sorted([int(x) for x in open('10.txt').read().strip().split('\n')])
    diff = [adapters[i+1] - adapters[i] for i in range(len(adapters) - 1)] + [3]
    print(diff.count(1)*diff.count(3))

    # try dynamic programming for part 2 (did not work, but would solve every input, not just the ones with at most four 1s in diff)
    diff_exclude = [[adapters[i+1] - adapters[i], adapters[i+2] - adapters[i], adapters[i+3] - adapters[i]] for i in range(len(adapters) - 3)]
    arangements_till = {0: 1}
    for i in range(1, len(diff_exclude)):
        x0 = diff_exclude[i - 1][0] if i > 0 else 9
        x1 = diff_exclude[i - 2][1] if i > 1 else 9
        x2 = diff_exclude[i - 3][2] if i > 2 else 9

        arag = 0
        if x0 < 4:
            arag += arangements_till[i - 1]
        if x1 < 4:
            arag += arangements_till[i - 2]
        if x2 < 4:
            arag += arangements_till[i - 3]
#        print(diff_exclude[i], x0, x1, x2, arag)
        arangements_till[i] = arag
#    print(arangements_till.values())
        

    # this assumes there are at most four 1s between neighboured 3s in diff
    # with only four 1s the number of possiblities to arrange them is simple
    # if there are four, one of all combination does not work (where every adapter is removed)
    # otherwise all combination 2 ** count do work
    sol = 1
    last3 = -1
    for i in range(len(diff)):
        if diff[i] == 3:
            nr1 = i - last3 - 1
            assert(nr1 <= 4)
            if nr1 > 3:
                sol *= 2 ** (nr1 - 1) - 1
            elif nr1 > 1:
                sol *= 2 ** (nr1 - 1)
            last3 = i
    print(sol)

def day11():
    originalSeats = [list(x) for x in open('11.txt').read().strip().split('\n')]
    rows, cols = len(originalSeats), len(originalSeats[0])
    directions = list(itertools.product(range(-1, 2), range(-1, 2)))
    directions.remove((0,0))

    # Part 1
    def neighboursOccupied(row, col, seats):
        return len(list(filter(lambda x: x == '#', [seats[x][y] if  0 <= x < rows and 0 <= y < cols else 'O' for x, y in [(row + x, col + y) for x, y in directions]])))

    seatscopy = []
    seats = deepcopy(originalSeats)
    while seats != seatscopy:
        seatscopy = deepcopy(seats)
        seats = [[ '.' if seatscopy[row][col] == '.' else 'L' if neighboursOccupied(row, col, seatscopy) >= 4 else '#' if neighboursOccupied(row, col, seatscopy) == 0 else seatscopy[row][col] for col in range(cols)] for row in range(rows)]

    print(len(list(filter(lambda x: x == '#', [seats[r][c] for r, c in itertools.product(range(rows), range(cols))]))))

    # Part 2
    def neighboursOccupied2(row, col, seats):
        res = 0
        for dirX, dirY in directions:
            x, y = dirX, dirY
            while 0 <= row + x < rows and 0 <= col + y < cols:
                if seats[row + x][col + y] == '#':
                    res += 1
                    break
                if seats[row + x][col + y] == 'L': break
                x += dirX
                y += dirY
        return res

    seatscopy = []
    seats = deepcopy(originalSeats)
    while seats != seatscopy:
        seatscopy = deepcopy(seats)
        seats = [[ '.' if seatscopy[row][col] == '.' else 'L' if neighboursOccupied2(row, col, seatscopy) >= 5 else '#' if neighboursOccupied2(row, col, seatscopy) == 0 else seatscopy[row][col] for col in range(cols)] for row in range(rows)]

    print(len(list(filter(lambda x: x == '#', [seats[r][c] for r, c in itertools.product(range(rows), range(cols))]))))

def day12():
    N = open('12.txt').read().strip().split('\n')
    gradM = {0: np.array([0,1]), 90: np.array([1,0]), 180: np.array([0,-1]), 270: np.array([-1,0]), }

    # Part 1
    pos = np.array([0, 0])
    dire = 90 # facing east

    for n in N:
        action = n[0]
        if action == 'N': pos += gradM[0]*int(n[1:])
        if action == 'E': pos += gradM[90]*int(n[1:])
        if action == 'S': pos += gradM[180]*int(n[1:])
        if action == 'W': pos += gradM[270]*int(n[1:])
        if action == 'R': dire = (dire + int(n[1:])) % 360
        if action == 'L': dire = (dire - int(n[1:])) % 360
        if action == 'F': pos += gradM[dire]*int(n[1:])

    print(abs(pos[0]) + abs(pos[1]))

    # Part 2
    transition = {0: np.array([[1,0], [0,1]]), 90: np.array([[0,1], [-1,0]]), 180: np.array([[-1,0], [0,-1]]), 270: np.array([[0,-1], [1,0]])}
    pos = np.array([0, 0])
    wayPos = np.array([10,1])

    for n in N:
        action = n[0]
        if action == 'N': wayPos += gradM[0]*int(n[1:])
        if action == 'E': wayPos += gradM[90]*int(n[1:])
        if action == 'S': wayPos += gradM[180]*int(n[1:])
        if action == 'W': wayPos += gradM[270]*int(n[1:])
        if action == 'R': wayPos = np.matmul(transition[int(n[1:])], wayPos)
        if action == 'L': wayPos = np.matmul(transition[360 - int(n[1:])], wayPos)
        if action == 'F': pos += wayPos*int(n[1:])

    print(abs(pos[0]) + abs(pos[1]))

def day13():
    inpu = open('13.txt').read().split('\n')
    arivalTime = int(inpu[0])
    busses = [i for i in inpu[1].split(',')]

    # Part 1
    wait = arivalTime
    bus = 0
    for b in busses:
        if b == 'x':
            continue
        b = int(b)
        waitTime = b - (arivalTime % b)
        if  waitTime < wait:
            wait = waitTime
            bus = b
    print(bus*wait)

    # Part 2
    # Note: the bus times are only prime numbers so they can easily multiplied to get the kgV
    # pre, inc = solve (i) finds the correct departure time for the first i busses.
    # For exactly all times 'pre + j*inc' the first i busses arrive at the correct time
    # To add the next bus j can be iterated until a time is found, where bus i+1 leaves at the correct time
    # As inc is large the number of iterations is minimal
    # The new pre is then pre + j*inc and the new increment is then kgV of (inc, busTime)
    def solve(i):
        if i == 0:
            return int(busses[i]), int(busses[i])
        if busses[i] == 'x':
            return solve(i - 1)
        departure, inc = solve(i - 1)
        b = int(busses[i])
        while (departure + i) % b != 0:
            print(departure)
            departure += inc
        return departure, inc * b # cause of the prime numbers inc * b is correct

    print(solve(len(busses) - 1)[0])

def day14():
    P = [x.split(' ') for x in open('14.txt').read().strip().split('\n')]

    # Part 1
    memory = {}
    for p in P:
        if p[0] == 'mask':
            mask = p[2]
            continue
        address = p[0][4:-1]
        v = int(p[2])
        v_res = 0
        for i, x in enumerate(reversed(mask)):
            if x == '1': v_res += 2 ** i
            elif x == 'X': v_res += 0 if (v >> i) % 2 == 0 else 2 ** i
        memory[address] = v_res
    print(sum(memory.values()))

    # Part 2
    memory = {}
    for p in P:
        if p[0] == 'mask':
            mask = p[2]
            continue
        address = int(p[0][4:-1])
        a_res = ''
        for i, x in enumerate(reversed(mask)):
            if x == '0': a_res += str((address >> i) % 2)
            else: a_res += x

        addresses_x1 = [a_res]
        while addresses_x1[0].count('X') > 0:
            addresses_x2 = []
            for a in addresses_x1:
                addresses_x2.append(a.replace('X', '0', 1))
                addresses_x2.append(a.replace('X', '1', 1))
            addresses_x1 = addresses_x2

        for a in addresses_x1:
            memory[a] = int(p[2])
    print(sum(memory.values()))

def day15():
    N = [int(x) for x in open('15.txt').read().split(',')]
    last = {} # map value to second last and last time it was spoken
    for i in range(2020):
#    for i in range(30000000):
        spoken = N[i] if i < len(N) else last[spoken][1] - last[spoken][0] if last[spoken][0] != None else 0
        last[spoken] = last[spoken][1] if spoken in last else None, i
    print(spoken)

def day16():
    I = open('16.txt').read().split('\n\n')
    rules = [x.split(': ') for x in I[0].strip().split('\n')]
    rules = list(map(lambda x: [x[0], [int(x[1].split('or')[a].split('-')[b]) for a, b in itertools.product(range(2), repeat=2)]], rules))
    your_ticket = [int(x) for x in I[1].split(':')[1].strip().split(',')]
    nearby_tickets = [list(map(int, x.split(','))) for x in I[2].split(':')[1].strip().split('\n')]

    # Part 1
    scan_errors = []
    incorrect_tickets = []
    for t in nearby_tickets:
        for val in t:
            ok = False
            for r in rules:
                if r[1][0] <= val <= r[1][1] or r[1][2] <= val <= r[1][3]:
                    ok = True
                    break
            if not ok:
                scan_errors.append(val)
                incorrect_tickets.append(t)

    print(sum(scan_errors))

    # Part 2
    correct_tickets = [t for t in nearby_tickets if t not in incorrect_tickets]
    # find which rules could fit on which position, there can be mulitple valid rules
    r_for_pos = defaultdict(list)
    for r in rules:
        for i in range(len(your_ticket)):
            # check if rule applies for ticket value i
            ok = True
            for t in correct_tickets:
                val = t[i]
                if r[1][0] <= val <= r[1][1] or r[1][2] <= val <= r[1][3]:
                    continue
                ok = False
                break
            if ok:
                r_for_pos[i].append(r[0])

    # find the exact rule for each position. If a position only allows one rule, this is the one.
    # Remove this rule from all other position and repeat
    r_to_i = {}
    while r_for_pos:
        x = [(i, r_for_pos[i][0]) for i in r_for_pos if len(r_for_pos[i]) == 1]
        i, r = x[0]
        r_to_i[r] = i
        r_for_pos.pop(i)
        for j in r_for_pos:
            if r in r_for_pos[j]:
                r_for_pos[j].remove(r)

    result = [your_ticket[i] for i in [r_to_i[x] for x in r_to_i if x.startswith('departure')]]
    ans = 1
    for res in result:
        ans *= res
    print(ans)

def day18():
    E = [[l for l in list(x) if l != ' '] for x in open('18.txt').read().strip().split('\n')]

    # Part 1
    def evaluate(e):
        if len(e) == 1: return int(e[0])
        if '(' in e:
            for i, v in enumerate(e):
                if v == '(': startpar = i
                elif v == ')':
                    endpar = i
                    p_result = evaluate(e[startpar + 1:endpar])
                    break
            return evaluate(e[:startpar] + [p_result] + e[endpar + 1:])
        if e[1] == '+': return evaluate([int(e[0]) + int(e[2])] + e[3:])
        if e[1] == '*': return evaluate([int(e[0]) * int(e[2])] + e[3:])
    print(sum([evaluate(e) for e in E]))

    # Part 2
    def evaluate2(e):
        if len(e) == 1: return int(e[0])
        if '(' in e:
            for i, v in enumerate(e):
                if v == '(': startpar = i
                elif v == ')':
                    endpar = i
                    p_result = evaluate2(e[startpar + 1:endpar])
                    break
            return evaluate2(e[:startpar] + [p_result] + e[endpar + 1:])
        if '+' in e:
            i = e.index('+')
            return evaluate2(e[:i - 1] + [int(e[i - 1]) + int(e[i + 1])] + e[i + 2:])
        if e[1] == '*': return evaluate2([int(e[0]) * int(e[2])] + e[3:])
    print(sum([evaluate2(e) for e in E]))

def day19():
    inputFile = "19.txt"
    G_original = {x.split(':')[0]: [y.strip().split(' ') for y in x.split(':')[1].split('|')] for x in open(inputFile).read().split('\n\n')[0].split('\n')}
    I = ([x for x in open(inputFile).read().split('\n\n')[1].strip().split('\n')])

    # Part 1
    # Goal is to match a context free grammar without loops (so the language is acutally finite and thus regular)
    # First naive approach find all words in the language
    # if rule = '0' the whole language will be the result
    def matchingStrings(rule):
        if (rule == '"a"'): return {'a'}
        if (rule == '"b"'): return {'b'}

        res = set()
        for r in G_original[rule]:
            submatches = [matchingStrings(x) for x in r]
            if len(submatches) == 1:
                res.update(submatches[0])
                continue
            res.update({''.join(y) for y in itertools.product(*submatches)})
        return res

    # Part 2
    # solution of Part 1 will not work, as the language is now infinite
    # Use CYK algorithm which is in O(n^3) where n is the length of the word to test
    # The cache improves the runtime hugely
    # However the Grammar needs to be in Chomsky Normal Form (CNF)
    G = deepcopy(G_original)
    G['8'] = [['42'], ['42', '8']]
    G['11'] = [['42', '31'], ['42', '11', '31']] # normalize to CNF
    G['11'] = [['42', '31'], ['11_0', '31']]
    G['11_0'] = [['42', '11']]

    @functools.cache
    def CYK(word): # G is not an argument, as it cannot be hashed for the cache
        print(word)
        if len(word) < 1:
            return set()
        if len(word) == 1:
            return {x for x in G if ['"' + word + '"'] in G[x] }
        result = set()
        for i in range(1, len(word)):
            A = CYK(word[:i])
            B = CYK(word[i:])
            rhs = list(itertools.product(A, B))
            for p in G:
                for r in rhs:
                    if list(r) in G[p]:
                        result.add(p)
        return result

    def toCNF(grammar):
        res = {}
        for p in grammar:
            res[p] = []
            for r in grammar[p]:
                if len(r) == 2:
                    res[p].append(r)
                elif len(r) == 1:
                    if r[0] == '"a"' or r[0] == '"b"':
                        res[p].append(r)
                    else:
                        # assume grammar[r[0]] has len 2
                        res[p] += grammar[r[0]]
                else: # len > 2
                    assert(False)
        return res

    G = toCNF(G)
    cor = list(filter(lambda x: '0' in CYK(x), (w for w in I)))
    L = matchingStrings('0')

    print("Part1:", len([x for x in I if x in L]))
    print("Part2:", len(cor))

def day23():
    LABELS = [int(x) for x in open("23.txt").read().strip()]

    # Use a linked list to handle the circle. 
    # Then only the reference needs to be update on each step
    # and not the memory copied
    class list_item:
        def __init__(self, value, next_item):
            self.value = value
            self.next_item = next_item

        def __str__(self):
            return str(list(self))

        def __iter__(self):
            class iter:
                def __init__(self, start):
                    self.start = start
                    self.current = start
                    self.counter = 0

                def __next__(self):
                    if (self.counter > 0 and self.current == self.start):
                        raise StopIteration
                    value = self.current.value
                    self.counter += 1
                    self.current = self.current.next_item
                    return value
            return iter(self)

    def shuffle(circle, times):
        l = len(circle)

        # create the linked list
        items = {}
        last_item = None
        for cup in reversed(circle):
            last_item = list_item(cup, last_item)
            items[cup] = last_item

        # close the loop from the last to the first item
        items[circle[l - 1]].next_item = items[circle[0]]
        items = tuple(items[i + 1] for i in range(len(circle)))

        # next step
        cur = items[circle[0] - 1]
        for i in range(times):
            if i % 10000 == 0:
                print(i)
            pick_it = cur
            pick_values = []
            for i in range(3):
                pick_it = pick_it.next_item
                pick_values.append(pick_it.value)
            after_value = (cur.value - 2) % l + 1
            while after_value in pick_values: after_value = (after_value - 2) % l + 1
            after_item = items[after_value - 1]
            # bend the new next pointers
            start_pick = cur.next_item
            cur.next_item = pick_it.next_item
            pick_it.next_item = after_item.next_item
            after_item.next_item = start_pick
            cur = cur.next_item

        return items

    # Part 1
    items = shuffle(LABELS, 100)
    res = ''.join([str(x) for x in items[0]])
    print(res[1:])

    # Part 2
    items = shuffle(LABELS + [x for x in range(10,10 ** 6 + 1)], 10 ** 7)
    n = items[0].next_item
    nn = n.next_item
    print(n.value * nn.value)

day23()
