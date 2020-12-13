#!/usr/bin/env python3

import sys
import re
import functools
import operator
import itertools
import numpy as np
from pprint import pprint
from copy import deepcopy

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

day13()
