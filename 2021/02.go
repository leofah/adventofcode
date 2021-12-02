package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

func day2() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Split(bufio.ScanWords)
	forward := 0
	depth1 := 0
	depth2 := 0
	aim := 0
	for scanner.Scan() {
		dir := scanner.Text()
		scanner.Scan()
		count, _ := strconv.Atoi(scanner.Text())
		switch dir[0] {
		case 'f':
			forward += count
			depth2 += aim * count
			break
		case 'd':
			depth1 += count
			aim += count
			break
		case 'u':
			depth1 -= count
			aim -= count
			break
		}
	}
	fmt.Println(forward * depth1)
	fmt.Println(forward * depth2)
}

func main() {
	day2()
}
