package com.sjq.algorithm;


/**
 * @Author Kemp, 前缀树实现
 * @create 2024/5/13 12:28
 */
class Trie {
    TreeNode root;

    public Trie() {
        root = new TreeNode();
    }

    // 插入字符串，遍历每个字符，如果对应字符的子节点不存在，则创建
    public void insert(String word) {
        TreeNode node = root;
        for (char ch : word.toCharArray()) {
            int chrIdx = ch - 'a';
            if (node.childs[chrIdx] == null) {
                node.childs[chrIdx] = new TreeNode();
            }
            node = node.childs[chrIdx];
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        TreeNode node = searchPreFix(word);
        return node != null && node.isEnd;
    }

    public boolean startsWith(String prefix) {
        return searchPreFix(prefix) != null;
    }

    // 查找前缀，按字符进行遍历，遇到null就返回
    private TreeNode searchPreFix(String prefix) {
        TreeNode node = root;
        for (char ch : prefix.toCharArray()) {
            int chrIdx = ch - 'a';
            if (node.childs[chrIdx] == null) {
                return null;
            }
            node = node.childs[chrIdx];
        }
        return node;
    }


    private class TreeNode {
        boolean isEnd;
        TreeNode[] childs;

        TreeNode() {
            childs = new TreeNode[26];
        }
    }
}

