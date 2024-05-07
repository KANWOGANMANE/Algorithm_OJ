package com.sjq.algorithm;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

/**
 * @Author Kemp
 * @create 2024/5/4 11:40
 */
public class MinStack {
    Deque<List<Integer>> stack;
    long lastMin;

    public MinStack() {
        stack = new LinkedList<>();
        lastMin = Long.MAX_VALUE;
    }

    public void push(int val) {
        // 记录每个栈元素当前的最小值
        List<Integer> list = new ArrayList<>(2);
        list.add(val);
        list.add(Math.min(val, getMin()));
        stack.push(list);
    }

    public void pop() {
        stack.pop();
    }

    public int top() {
        return stack.peek().get(0);
    }

    public int getMin() {
        return !stack.isEmpty() ? stack.peek().get(1) : Integer.MAX_VALUE;
    }
}
