package main

import (
	"bufio"
	"container/list"
	"fmt"
	"os"
	"strconv"
)

func day1() {
	scanner := bufio.NewScanner(os.Stdin)
	oldVal := 0
	count1 := -1
	count2 := 0
	queue := list.New()
	for scanner.Scan() {
		inp := scanner.Text()
		val, _ := strconv.Atoi(inp)
		//1
		if val > oldVal {
			count1++
		}
		oldVal = val
		//2
		queue.PushBack(val)
		if queue.Len() == 4 {
			fron := queue.Front()
			queue.Remove(fron)
			oldVal := fron.Value.(int)
			if val > oldVal {
				count2++
			}
		}
	}
	//fmt.Printf("%d\n", count1)
	fmt.Printf("%d\n", count2)
}

func main() {
	day1()
}
