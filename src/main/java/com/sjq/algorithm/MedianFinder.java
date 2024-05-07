package com.sjq.algorithm;

import java.util.PriorityQueue;

/**
 * @Author Kemp
 * @create 2024/5/3 13:05
 */
public class MedianFinder {
    // 大顶堆，小于中位数
    PriorityQueue<Integer> bigHeap;
    // 小顶堆，大于等于中位数
    PriorityQueue<Integer> smallHeap;


    public MedianFinder() {
        bigHeap = new PriorityQueue<>((o1, o2) -> Integer.compare(o2, o1));
        smallHeap = new PriorityQueue<>(Integer::compare);
    }

    public void addNum(int num) {
        if (bigHeap.isEmpty() && smallHeap.isEmpty()) {
            smallHeap.add(num);
            return;
        }
        double midNUm = findMedian();
        int size = bigHeap.size() + smallHeap.size();
        // 大顶堆(小于中位数)   中位数  小顶堆（大于等于中位数）
        // num>中位，放小顶堆，如果当前是偶数，小顶堆pok，放到大顶堆，
        if ((size & 1) == 1) {
            // 当前是奇数个，再加1等于偶数个，中位数取，大小顶堆的头/ 2
            if (num >= midNUm) {
                smallHeap.add(num);
                bigHeap.add(smallHeap.poll());
            } else {
                bigHeap.add(num);
            }
        } else {
            // 当前是偶数个，在加1个等于奇数个，中位数从小顶堆取
            if (num >= midNUm) {
                smallHeap.add(num);
            } else {
                bigHeap.add(num);
                smallHeap.add(bigHeap.poll());
            }
        }
    }

    public double findMedian() {
        if (bigHeap.size() == smallHeap.size()) {
            return (bigHeap.peek() + smallHeap.peek()) / 2.0;
        } else {
            return smallHeap.peek();
        }
    }
}
