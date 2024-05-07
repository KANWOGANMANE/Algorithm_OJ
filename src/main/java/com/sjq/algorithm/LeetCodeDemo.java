package com.sjq.algorithm;

import com.sjq.model.TreeNode;

import java.util.*;

/**
 * @Author Kemp
 * @create 2024/3/2 14:06
 */
public class LeetCodeDemo {
    public static void main(String[] args) {
        List<Integer> anagrams = findAnagrams("abab", "ab");
        int subarraySum = subarraySum(new int[]{1, 2, 3, 4, 5}, 7);
        int[] maxSlidingWindow = maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3);
        String minWindow = minWindow("ADOBECODEBANC", "ABC");
        int subArray = maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4});
        long maxSubArray = maxSubArray(new long[]{-2, 1, -3, 4, -1, 2, 1, -5, 4});
        int missingPositive = firstMissingPositive(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 20});
        setZeroes(new int[][]{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}});
        List<Integer> spiralOrder = spiralOrder(new int[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        rotate(new int[][]{{15, 13, 2, 5}, {14, 3, 4, 1}, {12, 6, 8, 9}, {16, 7, 10, 11}});
        List<String> parenthesis = generateParenthesis(2);
        int maxProfit = maxProfit(new int[]{1, 3, 7, 6, 5, 9});
        List<List<String>> partition = partition("");
        List<List<Integer>> subsets = subsets(new int[]{1, 2, 3});
        int numSquare = numSquares(13);
        int coinNums = coinChange(new int[]{1, 2, 5}, 11);
        int uniquePaths = uniquePaths(3, 7);
        int minPathSum = minPathSum(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        buildTree(new int[]{3, 9, 20, 15, 7}, new int[]{9, 3, 15, 20, 7});
        TreeNode flattenTreeNode = new TreeNode(1, new TreeNode(2, new TreeNode(3), new TreeNode(4)), new TreeNode(5, new TreeNode(7), new TreeNode(6)));
        TreeNode node = new TreeNode(1, new TreeNode(2, new TreeNode(3), null), null);
        TreeNode node2 = new TreeNode(1, new TreeNode(2, new TreeNode(3, new TreeNode(5), null), new TreeNode(4)), null);
        flatten(flattenTreeNode);
        longestValidParentheses(")()()(()))");
        minDistance("", "");
        findKthLargest(new int[]{3, 2, 1, 5, 6, 4}, 2);
        topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 2);
        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);
        medianFinder.addNum(3);
        medianFinder.addNum(4);
        medianFinder.addNum(5);
        medianFinder.addNum(2);
        double median = medianFinder.findMedian();
        MinStack minStack = new MinStack();
        minStack.push(-2);
        minStack.push(0);
        minStack.push(-3);
        int stackMin1 = minStack.getMin();
        minStack.pop();
        minStack.top();
        int stackMin2 = minStack.getMin();
        twoSum(new int[]{2, 7, 11, 15}, 9);
        String decodeString = decodeString("3[z]2[2[y]pq4[2[jk]e1[f]]]ef");
        int searchInsert = searchInsert(new int[]{1, 3, 5, 6}, 5);
        boolean searchMatrix = searchMatrix(new int[][]{{1}, {3}}, 3);
        int searchLeetCode33 = searchLeetCode33(new int[]{3, 5, 1}, 3);
    }

    public static int searchLeetCode33(int[] nums, int target) {
        int index = -1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] < 0) {
                index = i;
            }
        }

        if (index == -1) {
            return searchInsert(0, nums.length - 1, nums, target);
        }

        if (target > nums[nums.length - 1]) {
            return searchInsert(0, index - 1, nums, target);
        } else {
            return searchInsert(index, nums.length - 1, nums, target);
        }
    }

    public static boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length - 1;
        int left = 0, right = m - 1, mid = (right - left) / 2 + left;
        if (m < 2) {
            return search(matrix[0], target);
        }

        while (left <= right) {
            if (matrix[mid][0] == target) {
                return true;
            }
            if (n != 0 && mid - 1 >= 0 && target >= matrix[mid - 1][0] && target < matrix[mid][0]) {
                return search(matrix[mid - 1], target);
            }
            if (n != 0 && target > matrix[mid][n] && mid + 1 < m && target <= matrix[mid + 1][n]) {
                return search(matrix[mid + 1], target);
            }

            if (target > matrix[mid][0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
            mid = (right - left) / 2 + left;
        }
        return false;
    }

    public static boolean search(int[] matrix, int target) {
        int left = 0, right = matrix.length - 1;

        // 该计算方式相当于(left + right) / 2 ，目的是为了防止(left + right)溢出
        int mid = (right - left) / 2 + left;
        while (left <= right) {
            if (matrix[mid] == target) {
                return true;
            }
            if (target > matrix[mid]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
            mid = (right - left) / 2 + left;
        }
        return false;
    }

    public static int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return left;
    }

    public static int searchInsert(int left, int right, int[] nums, int target) {
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    public static String decodeString(String s) {
        Deque<String> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if ("]".equals(String.valueOf(s.charAt(i)))) {
                StringBuilder sb = new StringBuilder();
                while (!stack.isEmpty() && !"[".equals(stack.peek())) {
                    sb.insert(0, stack.pop());
                }
                stack.pop();

                StringBuilder sb2 = new StringBuilder();
                while (!stack.isEmpty() && Character.isDigit(stack.peek().charAt(0))) {
                    sb2.insert(0, stack.pop());
                }

                int time = Integer.parseInt(sb2.toString());
                StringBuilder sb3 = new StringBuilder();
                for (int j = 0; j < time; j++) {
                    sb3.append(sb);
                }
                stack.push(sb3.toString());
            } else {
                stack.push(String.valueOf(s.charAt(i)));
            }
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.insert(0, stack.pop());
        }
        return sb.toString();
    }

    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            if (hashtable.containsKey(target - nums[i])) {
                return new int[]{hashtable.get(target - nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }

    public static int[] topKFrequent(int[] nums, int k) {
        // 先通过map统计频率，然后通过堆排序取频率最高的k个数
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        PriorityQueue<List<Integer>> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o.get(1)));
        int size = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (size < k) {
                priorityQueue.add(Arrays.asList(entry.getKey(), entry.getValue()));
            } else if (priorityQueue.peek().get(1) < entry.getValue()) {
                priorityQueue.poll();
                priorityQueue.add(Arrays.asList(entry.getKey(), entry.getValue()));
            }
            size++;
        }
        return priorityQueue.stream().mapToInt(it -> it.get(0)).toArray();
    }

    public static int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
        for (int i = 0; i < nums.length; i++) {
            if (i < k) {
                priorityQueue.add(nums[i]);
            } else if (priorityQueue.peek() < nums[i]) {
                priorityQueue.poll();
                priorityQueue.add(nums[i]);
            }
        }

        return priorityQueue.poll();
    }

    public static int minDistance(String word1, String word2) {
        /*
          '' a b c
       '' 0  1 2 3
        b 1  1
        c 2
        d 3
         */
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= word2.length(); i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
            }
        }

        return dp[word1.length()][word2.length()];
    }

    public static int longestValidParentheses(String s) {
        // 定义dp[i]为，以i为结尾的字符串，最长的有效字符串数量
        Stack<String> stack = new Stack<>();
        int[] dp = new int[s.length()];
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            String item = String.valueOf(s.charAt(i));
            if ("(".equals(item)) {
                stack.push(item);
            } else {
                if (!stack.isEmpty()) {
                    stack.pop();
                    dp[i] = dp[i - 1] + 1;
                    int lastIndex = i - dp[i] * 2;
                    if (lastIndex > 0) {
                        dp[i] += dp[lastIndex];
                    }
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res * 2;
    }

    public static void flatten(TreeNode root) {
        flattenTreeNode(root);
    }

    public static TreeNode flattenTreeNode(TreeNode root) {
        // 返回值为该节点拍平后的最后一个节点
        if (root == null) {
            return null;
        }

        TreeNode left = root.left;
        TreeNode right = root.right;
        TreeNode flattenLeft = flattenTreeNode(left);
        TreeNode flattenRight = flattenTreeNode(right);
        if (flattenLeft != null) {
            root.right = root.left;
            root.left = null;
            if (flattenRight != null) {
                flattenLeft.right = right;
                return flattenRight;
            } else {
                return flattenLeft;
            }
        } else {
            if (flattenRight != null) {
                return flattenRight;
            } else {
                return root;
            }
        }
    }

    public TreeNode process(TreeNode root) {
        if (root == null) {
            return null;
        }
        // 将左节点变为链表，并且返回列表的最后一个元素
        TreeNode leftProcess = process(root.left);
        // 将右节点变为链表，并且返回列表的最后一个元素
        TreeNode rightProcess = process(root.right);
        // 拼接左右列表
        if (leftProcess != null) {
            TreeNode temp = root.right;
            root.right = root.left;
            root.left = null;
            if (rightProcess != null) {
                leftProcess.right = temp;
                return rightProcess;
            } else {
                return leftProcess;
            }
        } else {
            if (rightProcess != null) {
                return rightProcess;
            } else {
                return root;
            }
        }
    }

    public static void buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        buildTreeFromPreAndInOrder(map, preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }

    public static TreeNode buildTreeFromPreAndInOrder(Map<Integer, Integer> map, int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }
        // pre 3,9,20,15,7
        // in 9,3,15,20,7
        TreeNode root = new TreeNode(preorder[preStart]);
        Integer inorderRoot = map.get(preorder[preStart]);
        int leftTreeSize = inorderRoot - inStart;
        root.left = buildTreeFromPreAndInOrder(map, preorder, preStart + 1, preStart + leftTreeSize, inorder, inStart,
                inorderRoot - 1);
        root.right = buildTreeFromPreAndInOrder(map, preorder, preStart + leftTreeSize + 1, preEnd, inorder,
                inorderRoot + 1, inEnd);
        return root;
    }

    public static int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int sum = 0;
        for (int i = 0; i < m; i++) {
            sum += grid[i][0];
            grid[i][0] = sum;
        }
        sum = 0;
        for (int i = 0; i < n; i++) {
            sum += grid[0][i];
            grid[0][i] = sum;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                grid[i][j] = Math.min(grid[i - 1][j] + grid[i][j], grid[i][j - 1] + grid[i][j]);
            }
        }
        return grid[m - 1][n - 1];
    }

    public static int uniquePaths(int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 1;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }

    public static int coinChange(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }
        int[] conArray = new int[amount + 1];
        Arrays.fill(conArray, -1);
        conArray[0] = 0;
        for (int coin : coins) {
            if (coin < conArray.length) {
                conArray[coin] = 1;
            }
        }

        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i && conArray[i - coin] != -1) {
                    if (conArray[i] == -1 || conArray[i] > conArray[i - coin] + 1) {
                        conArray[i] = conArray[i - coin] + 1;
                    }
                }
            }
        }
        return conArray[amount];
    }

    public static int numSquares(int n) {
        int max = Integer.MAX_VALUE;
        int[] dp = new int[n + 1];
        // 初始化
        for (int j = 0; j <= n; j++) {
            dp[j] = max;
        }
        // 当和为0时，组合的个数为0
        dp[0] = 0;
        // 遍历背包
        for (int j = 1; j <= n; j++) {
            // 遍历物品
            for (int i = 1; i * i <= j; i++) {
                int index = j - i * i;
                dp[j] = Math.min(dp[j], dp[index] + 1);
            }
        }
        return dp[n];
    }

    public static List<List<String>> partition(String s) {
        return new ArrayList<>();
    }

    static List<List<Integer>> resSubsets = new ArrayList<>();
    static List<Integer> ansSubsets = new ArrayList<>();

    public static List<List<Integer>> subsets(int[] nums) {
        dfsSubsets(nums, 0);
        return resSubsets;
    }

    public static void dfsSubsets(int[] nums, int index) {
        if (index == nums.length) {
            resSubsets.add(new ArrayList<>(ansSubsets));
            return;
        }

        ansSubsets.add(nums[index]);
        dfsSubsets(nums, index + 1);
        ansSubsets.remove(ansSubsets.size() - 1);
        dfsSubsets(nums, index + 1);
    }

    public static int maxProfit(int[] prices) {
        int buyPrice = Integer.MAX_VALUE;
        int profit = 0;
        for (int price : prices) {
            buyPrice = Math.min(buyPrice, price);
            profit = Math.max(profit, price - buyPrice);
        }
        return profit;
    }

    public static List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        addAllParenthesis(n, 0, 0, res, "");
        return res;
    }

    private static void addAllParenthesis(int n, int left, int right, List<String> res, String tmpStr) {
        if (right > left) {
            return;
        }

        if (n == left && left == right) {
            res.add(tmpStr);
            return;
        }

        if (left < n) {
            addAllParenthesis(n, left + 1, right, res, tmpStr + "(");
        }

        if (left > right) {
            addAllParenthesis(n, left, right + 1, res, tmpStr + ")");
        }
    }

    public static void rotate(int[][] matrix) {
        int n = matrix.length;
        int matNums = n * n;

        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;

        Deque<Integer> deque = new LinkedList<>();
        while (matNums >= 1) {
            // 上
            for (int i = left; i <= right; i++) {
                deque.addFirst(matrix[top][i]);
            }
            top++;
            for (int i = top; i <= bottom; i++) {
                deque.addFirst(matrix[i][right]);
            }
            right--;
            for (int i = right; i >= left; i--) {
                deque.addFirst(matrix[bottom][i]);
            }
            bottom--;
            for (int i = bottom; i >= top; i--) {
                deque.addFirst(matrix[i][left]);
            }
            left++;

            top--;
            right++;
            bottom++;
            left--;

            for (int i = top; i <= bottom && matNums >= 1; i++) {
                matrix[i][right] = deque.removeLast();
                matNums--;
            }
            right--;

            //

            for (int i = right; i >= left && matNums >= 1; i--) {
                matrix[bottom][i] = deque.removeLast();
                matNums--;
            }
            bottom--;

            //

            for (int i = bottom; i >= top && matNums >= 1; i--) {
                matrix[i][left] = deque.removeLast();
                matNums--;
            }
            left++;

            //
            for (int i = left; i <= right && matNums >= 1; i++) {
                matrix[top][i] = deque.removeLast();
                matNums--;
            }
            top++;

        }
    }

    public static List<Integer> spiralOrder(int[][] matrix) {
        LinkedList<Integer> result = new LinkedList<>();
        int top = 0;
        int right = matrix[0].length - 1;
        int left = 0;
        int bottom = matrix.length - 1;
        int flag = matrix.length * matrix[0].length;
        while (flag >= 1) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
                flag--;
            }
            top++;
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
                flag--;
            }
            right--;
            for (int i = right; i >= left; i--) {
                result.add(matrix[bottom][i]);
                flag--;
            }
            bottom--;
            for (int i = bottom; i >= top; i--) {
                result.add(matrix[i][left]);
                flag--;
            }
            left++;
        }

        return result;
    }

    public static void setZeroes(int[][] matrix) {
        int yLen = matrix.length;
        int xLen = matrix[0].length;

        List<String> list = new ArrayList<>();
        for (int i = 0; i < matrix.length; i++) {
            for (int i1 = 0; i1 < matrix[i].length; i1++) {
                if (matrix[i][i1] == 0) {
                    matrix[0][i1] = 0;
                    matrix[i][i1] = 0;
                }
            }
        }
    }

    private static void updateZero(int[][] matrix, int i, int j) {
        for (int i1 = 0; i1 < matrix.length; i1++) {
            for (int i2 = 0; i2 < matrix[i1].length; i2++) {
                if (i1 == i) {
                    matrix[i1][i2] = 0;
                }

                if (i2 == j) {
                    matrix[i1][i2] = 0;
                }
            }
        }
    }

    public static int firstMissingPositive(int[] nums) {
        Arrays.sort(nums);

        int maxNum = nums[nums.length - 1];
        if (maxNum < 0 || nums[0] > 1) {
            return 1;
        }

        int maxLeft = Integer.MAX_VALUE;
        int min_plus = Integer.MAX_VALUE;
        int val = Integer.MAX_VALUE;

        boolean isSerials = Boolean.TRUE;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i - 1] > 0) {
                int n = nums[i] - nums[i - 1];
                if (n > 1) {
                    val = Math.min(val, nums[i - 1]);

                    if (n < maxLeft) {
                        maxLeft = n;
                    }
                }

                min_plus = Math.min(min_plus, nums[i - 1]);
            }

            if (Math.abs(nums[i] - nums[i - 1]) != 1) {
                isSerials = Boolean.FALSE;
            }
        }
        if (isSerials) {
            return maxNum + 1;
        }

        if (min_plus > 1) {
            return 1;
        }

        return val == Integer.MAX_VALUE ? min_plus + 1 : val + 1;
    }

    public static int maxSubArray(int[] nums) {
        if (nums.length == 1) return nums[0];

        // 动态规划，维护数组pre[i]，pre[i]意味已i为结尾，最大的连续子数组值为pre[i]
        int[] pre = new int[nums.length];
        pre[0] = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            pre[i] = Math.max(nums[i], pre[i - 1] + nums[i]);
            max = Math.max(max, pre[i]);
        }
        return max;
    }

    public static long maxSubArray(long[] nums) {
        if (nums.length == 1) return nums[0];

        // 贪心算法，连续和为负数，直接抛弃
        long max = Integer.MIN_VALUE;
        long sum = 0;
        for (long num : nums) {
            if (sum < 0) {
                sum = num;
            } else {
                sum += num;
            }
            max = Math.max(max, sum);
        }

        return max;
    }

    public static String minWindow(String s, String t) {
        Map<Character, Integer> tMap = new HashMap<>();
        Map<Character, Integer> sMap = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            tMap.put(t.charAt(i), tMap.getOrDefault(t.charAt(i), 0) + 1);
        }

        int sCount = 0;
        int minLen = Integer.MAX_VALUE;

        int strLeft = 0;

        int left = 0;
        int right = 0;
        while (right < s.length()) {
            char rightChar = s.charAt(right);
            right++;

            if (tMap.containsKey(rightChar)) {
                sMap.put(rightChar, sMap.getOrDefault(rightChar, 0) + 1);

                if (tMap.get(rightChar).equals(sMap.get(rightChar))) {
                    sCount++;
                }
            }

            while (sCount == tMap.size()) {
                if (right - left < minLen) {
                    minLen = right - left;
                    strLeft = left;
                }

                char leftChar = s.charAt(left);
                left++;

                if (tMap.containsKey(leftChar)) {
                    if (sMap.get(leftChar).equals(tMap.get(leftChar))) {
                        sCount--;
                    }
                    sMap.put(leftChar, sMap.getOrDefault(leftChar, 0) - 1);
                }
            }
        }

        return minLen == Integer.MAX_VALUE ? "" : s.substring(strLeft, strLeft + minLen);
    }

    public static int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && nums[i] > nums[deque.getFirst()]) {
                deque.removeFirst();
            }
            deque.addFirst(i);
        }

        res[0] = nums[deque.getLast()];
        for (int i = k; i < nums.length; i++) {
            while (!deque.isEmpty() && nums[i] > nums[deque.getFirst()]) {
                deque.removeFirst();
            }
            deque.addFirst(i);

            // 超过生命周期，移除
            if (deque.getLast() <= i - k) {
                deque.removeLast();
            }


            res[i - k + 1] = nums[deque.getLast()];
        }

        return res;
    }

    public static int subarraySum(int[] nums, int k) {
        /*
            1 2 3 4 5  ,   7
     nums   0 1 2 3 4
     pres   1 3 6 10 15
        nums[2] + nums[3]  == k
        pres[3] - pres[2]  == k

        ps:if: pres[3] - k == pres[2]
         */
        int count = 0;
        if (nums.length == 0) return count;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0;
        for (int num : nums) {
            sum += num;
            if (map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return count;
    }

    public static List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length()) return Collections.emptyList();
        int[] count = new int[26];
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < p.length(); i++) {
            int num = p.charAt(i) - 'a';
            count[num] += 1;
        }

        for (int l = 0, r = p.length(); r <= s.length(); l++, r++) {
            String substring = s.substring(l, r);
            int[] tmpCount = new int[26];
            for (int i = 0; i < substring.length(); i++) {
                int tmpNum = substring.charAt(i) - 'a';
                tmpCount[tmpNum] += 1;
            }
            if (Arrays.toString(count).equals(Arrays.toString(tmpCount))) {
                res.add(l);
            }
        }
        return res;
    }
}
