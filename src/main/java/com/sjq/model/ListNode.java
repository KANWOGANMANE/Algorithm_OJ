package com.sjq.model;

/**
 * @Author Kemp
 * @create 2024/5/18 13:16
 */
public class ListNode {
    public int val;
    public ListNode next;

    public ListNode(int val) {
        this.val = val;
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
