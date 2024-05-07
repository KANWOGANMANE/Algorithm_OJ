package com.sjq.algorithm;

import com.sjq.model.TreeNode;

import java.util.ArrayList;
import java.util.Stack;

/**
 * 1.平衡二叉树规定二叉树内不能有相同值
 * 2.父节点的左子树的所有节点必须比它小
 * 3.父节点的右子树的所有节点必须比它大
 *
 * @Author Kemp
 * @create 2023/12/1 23:14
 */
public class BalancedBinaryTreeUtil {
    /**
     * 平衡二叉树插入新的节点
     *
     * @param root  root
     * @param value value
     * @return root
     */
    public static TreeNode insert(TreeNode root, int value) {
        if (root == null) {
            return new TreeNode(value);
        }

        if (root.val < value) {
            root.right = insert(root.right, value);
        } else if (root.val > value) {
            root.left = insert(root.left, value);
        }

        return root;
    }

    /**
     * 查找平衡二叉树是否存在value
     *
     * @param root  root
     * @param value value
     * @return boolean
     */
    public static boolean isContain(TreeNode root, int value) {
        if (root == null) {
            return false;
        }
        if (root.val == value) {
            return true;
        } else if (root.val < value) {
            return isContain(root.right, value);
        } else {
            return isContain(root.left, value);
        }
    }

    /**
     * 1.删除的值没有子节点
     * 2.删除的值有1个子节点
     * 3.删除的值有2个子节点
     * （1）选择左子树最小那个作为节点，选择完后需要按平衡二叉树规则删除原节点
     * （2）现在右子树最大那个作为节点，选择完后需要按平衡二叉树规则删除原节点
     */
    public static void delete(TreeNode root, int value) {
        // todo
    }

    /**
     * 按层级返回二叉树
     *
     * @param root root
     * @return List
     */
    public static ArrayList<ArrayList<Integer>> getTreeList(TreeNode root) {
        //result用来存储结果
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        //创建两个辅助栈，分别存放奇数行和偶数行的节点
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();

        //创建集合，存放每一行的节点值
        ArrayList<Integer> list = new ArrayList<>();
        boolean flag = true;
        TreeNode node;
        stack1.push(root);
        while (!stack1.isEmpty() || !stack2.isEmpty()) {
            //奇数行，从左往右入栈stack2
            if (flag) {
                while (!stack1.isEmpty()) {
                    node = stack1.pop();
                    list.add(node.val);
                    if (node.left != null) {
                        stack2.push(node.left);
                    }
                    if (node.right != null) {
                        stack2.push(node.right);
                    }
                    if (stack1.isEmpty()) {
                        result.add(list);
                        list = new ArrayList<>();
                    }
                }
            } else {
                //偶数行，将入栈的奇数行出栈到stack1
                while (!stack2.isEmpty()) {
                    node = stack2.pop();//由于后进先出，所以弹出的是右子树
                    list.add(node.val);//将右节点存入
                    if (node.right != null) {
                        stack1.push(node.right);
                    }
                    if (node.left != null) {
                        stack1.push(node.left);
                    }
                    if (stack2.isEmpty()) {
                        result.add(list);
                        list = new ArrayList<>();
                    }
                }
            }
            flag = !flag;
        }
        return result;
    }

    /**
     * 打印二叉树到控制台
     *
     * @param root root
     */
    public static void print(TreeNode root) {
        if (root == null) System.out.println("EMPTY!");
        // 得到树的深度
        int treeDepth = getTreeDepth(root);

        // 最后一行的宽度为2的（n - 1）次方乘3，再加1
        // 作为整个二维数组的宽度
        int arrayHeight = treeDepth * 2 - 1;
        int arrayWidth = (2 << (treeDepth - 2)) * 3 + 1;
        // 用一个字符串数组来存储每个位置应显示的元素
        String[][] res = new String[arrayHeight][arrayWidth];
        // 对数组进行初始化，默认为一个空格
        for (int i = 0; i < arrayHeight; i++) {
            for (int j = 0; j < arrayWidth; j++) {
                res[i][j] = " ";
            }
        }

        // 从根节点开始，递归处理整个树
        // res[0][(arrayWidth + 1)/ 2] = (char)(root.val + '0');
        writeArray(root, 0, arrayWidth / 2, res, treeDepth);

        // 此时，已经将所有需要显示的元素储存到了二维数组中，将其拼接并打印即可
        for (String[] line : res) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < line.length; i++) {
                sb.append(line[i]);
                if (line[i].length() > 1 && i <= line.length - 1) {
                    i += line[i].length() > 4 ? 2 : line[i].length() - 1;
                }
            }
            System.out.println(sb.toString());
        }
    }

    /**
     * 获取二叉树深度
     *
     * @param root root
     * @return int
     */
    public static int getTreeDepth(TreeNode root) {
        return root == null ? 0 : (1 + Math.max(getTreeDepth(root.left), getTreeDepth(root.right)));
    }

    private static void writeArray(TreeNode currNode, int rowIndex, int columnIndex, String[][] res, int treeDepth) {
        // 保证输入的树不为空
        if (currNode == null) return;
        // 先将当前节点保存到二维数组中
        res[rowIndex][columnIndex] = String.valueOf(currNode.val);

        // 计算当前位于树的第几层
        int currLevel = ((rowIndex + 1) / 2);
        // 若到了最后一层，则返回
        if (currLevel == treeDepth) return;
        // 计算当前行到下一行，每个元素之间的间隔（下一行的列索引与当前元素的列索引之间的间隔）
        int gap = treeDepth - currLevel - 1;

        // 对左儿子进行判断，若有左儿子，则记录相应的"/"与左儿子的值
        if (currNode.left != null) {
            res[rowIndex + 1][columnIndex - gap] = "/";
            writeArray(currNode.left, rowIndex + 2, columnIndex - gap * 2, res, treeDepth);
        }

        // 对右儿子进行判断，若有右儿子，则记录相应的"\"与右儿子的值
        if (currNode.right != null) {
            res[rowIndex + 1][columnIndex + gap] = "\\";
            writeArray(currNode.right, rowIndex + 2, columnIndex + gap * 2, res, treeDepth);
        }
    }
}


