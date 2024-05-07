package com.sjq;

import com.alibaba.fastjson.JSONObject;
import com.sjq.algorithm.BalancedBinaryTreeUtil;
import com.sjq.model.TreeNode;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;

/**
 * @Author Kemp
 * @create 2023/12/2 0:03
 */
public class BalancedBinaryTreeUtilTest {

    @Test
    public void test_isContain() {
        TreeNode root = new TreeNode(10,
                new TreeNode(5,
                        new TreeNode(4), new TreeNode(6)),
                new TreeNode(20,
                        new TreeNode(19), new TreeNode(21)));

        boolean contain3 = BalancedBinaryTreeUtil.isContain(root, 19);
        Assert.assertTrue(contain3);

        boolean contain = BalancedBinaryTreeUtil.isContain(root, 21);
        Assert.assertTrue(contain);

        boolean contain1 = BalancedBinaryTreeUtil.isContain(root, 5);
        Assert.assertTrue(contain1);

        boolean contain2 = BalancedBinaryTreeUtil.isContain(root, 15);
        Assert.assertFalse(contain2);
    }

    @Test
    public void test_insert() {
        TreeNode root = new TreeNode(10,
                new TreeNode(5,
                        new TreeNode(4), new TreeNode(8)),
                new TreeNode(20,
                        new TreeNode(19), new TreeNode(21)));

        TreeNode insert = BalancedBinaryTreeUtil.insert(root, 18);
        TreeNode insert2 = BalancedBinaryTreeUtil.insert(root, 6);
        TreeNode insert3 = BalancedBinaryTreeUtil.insert(root, 5);

        BalancedBinaryTreeUtil.print(root);
        ArrayList<ArrayList<Integer>> treeList = BalancedBinaryTreeUtil.getTreeList(root);
        JSONObject.toJSONString(treeList);
    }
}
