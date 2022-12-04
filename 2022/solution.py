#!/usr/bin/env python3

def day1():
    elves = [sum([int(y) for y in x.split('\n')]) for x in open('01.txt').read().strip().split('\n\n')]
    print(max(elves))
    print(sum(sorted(elves)[-3:]))


def day2():
    input = [[x.split(" ")[0], x.split(" ")[1]] for x in open("02.txt").read().strip().split("\n")]
    score_selected = sum(ord(x) - 87 for _, x in input)
    score_won = 3 * sum((ord(y) - ord(x) + 2) % 3 for x, y in input)
    print(score_won + score_selected)
    score_won_2 = 3 * sum(ord(x) - 88 for _, x in input)
    score_selected_2 = sum((ord(y) + ord(x) - 1) % 3 + 1 for x, y in input)
    print(score_won_2 + score_selected_2)


def day3():
    from itertools import islice
    def prio(x): return ord(x) - (38 if ord(x) < ord('a') else 96)
    rucksacks = [[set(line[:(len(line)//2)]), set(line[len(line)//2:])] for line in open("03.txt").read().strip().split("\n")]
    both = [list(x.intersection(y))[0] for x, y in rucksacks]
    print(sum(prio(x) for x in both ))
    input = iter(line for line in open("03.txt").read().strip().split("\n"))
    groups = [[set(x) for x  in islice(input, 3)] for i in range(len(rucksacks)//3)]
    keys = [list(x.intersection(y).intersection(z))[0] for x, y, z in groups]
    print(sum(prio(x) for x in keys ))


if __name__ == "__main__":
    day3()
