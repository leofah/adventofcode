#!/usr/bin/env python3

def day1():
    elves = [sum([int(y) for y in x.split('\n')]) for x in open('01.txt').read().strip().split('\n\n')]
    print(max(elves))
    print(sum(sorted(elves)[-3:]))


if __name__ == "__main__":
    day1()
