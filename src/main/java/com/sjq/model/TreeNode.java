package com.sjq.model;

/**
 * @Author Kemp
 * @create 2023/12/1 23:09
 */
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode(int value) {
        this.val = value;
    }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
