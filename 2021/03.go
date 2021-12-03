package main

import (
	"bufio"
	"fmt"
	"os"
)

func day3() {
	scanner := bufio.NewScanner(os.Stdin)
	var count [12]int
	var size int
	for scanner.Scan() {
		dir := scanner.Text()
		size = len(dir)
		for i, c := range dir {
			if c == '1' {
				count[i] += 1
			} else {
				count[i] -= 1
			}
		}
	}
	var gamma int
	for i, b := range count {
		if b == 0 {
			continue //neither 0 or 1 are more common
		}
		if b < 0 {
			continue
		}
		gamma |= 1 << (size - i - 1)
	}
	epsilon := ^gamma & (1<<size - 1)
	fmt.Println(gamma * epsilon)
}

func main() {
	day3()
}
