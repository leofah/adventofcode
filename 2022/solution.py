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


if __name__ == "__main__":
    day2()
