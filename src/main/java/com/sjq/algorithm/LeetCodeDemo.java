package com.sjq.algorithm;

import com.sjq.model.ListNode;
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
        boolean searchMatrix = searchMatrix(new int[][]{
                {1, 4, 7, 11, 15},
                {2, 5, 8, 12, 19},
                {3, 6, 9, 16, 21},
                {10, 13, 14, 17, 24},
                {18, 21, 23, 26, 30}

        }, 20);
        int searchLeetCode33 = searchLeetCode33(new int[]{3, 5, 1}, 3);
        int min = findMin(new int[]{5, 6, 7, 1, 2, 3, 4});
        merge(new int[]{1, 2, 3, 0, 0, 0}, 3, new int[]{2, 5, 6}, 3);
        int numIslands = numIslands(new char[][]{
                {'1', '1', '1'},
                {'0', '1', '0'},
                {'1', '1', '1'}
        });
        int orangesRotting = orangesRotting(new int[][]{
                {2, 1, 1},
                {1, 1, 0},
                {0, 1, 1}
        });
        boolean canFinish = canFinish(6, new int[][]{
                {4, 0},
                {4, 1},
                {3, 1},
                {3, 2},
                {5, 4},
                {5, 3}
        });
        boolean canFinishDfs = canFinish(6, new int[][]{
                {4, 0},
                {4, 1},
                {3, 1},
                {3, 2},
                {5, 4},
                {5, 3}
        }, 1);
        Trie trie = new Trie();
        trie.insert("apple");
        trie.insert("appme");
        boolean hasApple = trie.search("apple");
        boolean hasApp = trie.startsWith("app");
        String longestWord = longestWord(new String[]{"a", "banana", "app", "appl", "ap", "apply", "apple"});
        int diameterOfBinaryTree = diameterOfBinaryTree(new TreeNode(1, new TreeNode(2, new TreeNode(4), new TreeNode(5)), new TreeNode(3)));
        boolean validBST = isValidBST(new TreeNode(5, new TreeNode(1), new TreeNode(10, new TreeNode(7, new TreeNode(6), new TreeNode(8)), new TreeNode(15, new TreeNode(14), new TreeNode(17)))));
        majorityElement(new int[]{2, 2, 1, 1, 1, 2, 2});
        sortColors(new int[]{2, 1, 1, 0, 2, 1, 0});
        ListNode listNode = mergeKLists(new ListNode[]{
                new ListNode(1, new ListNode(4, new ListNode(5))),
                new ListNode(2, new ListNode(4, new ListNode(6))),
                new ListNode(7, new ListNode(8, new ListNode(9))),
                new ListNode(7, new ListNode(9, new ListNode(10)))
        });
        int pathMaxSum = maxPathSum(new TreeNode(-10, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7))));
        int kthSmallest = kthSmallest(new TreeNode(3, new TreeNode(1, null, new TreeNode(2)), new TreeNode(4)), 1);
        double sortedArrays = findMedianSortedArrays(new int[]{1, 2, 3, 4}, new int[]{2, 3, 4, 5, 6});
        int duplicate = findDuplicate(new int[]{1, 3, 4, 2, 2});
        nextPermutation(new int[]{1, 2, 3});
        boolean unique = isUnique("abc");
    }

    public static boolean isUnique(String astr) {
        // 可以通过数组或者位运算进行判断
        if (astr.length() > 26) return false;
        // 通过位运算进行判断，左移index位
        int bits = 0;
        for (int i = 0; i < astr.length(); i++) {
            int index = astr.charAt(i) - 'a';
            if ((bits & (1 << index)) == 1) {
                return false;
            } else {
                bits = bits | (1 << index);
            }
        }
        return true;
    }

    public static void nextPermutation(int[] nums) {
        // 先从后往前找到降序的位置
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i >= 0) {
            // 当存在降序节点时，找到一个比nums[i]大一点点的数
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            // 交换位置
            swap(nums, i, j);
        }
        // 将i后面的数进行反转
        reverse(nums, i + 1);
    }

    public static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }

    public static int findDuplicate(int[] nums) {
        // 快慢指针，慢指针走1步，快指针走2步
        // 由于存在重复的数，则相当于存在环。
        int slow = 0;
        int fast = 0;
        // 寻找环的位置
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        // 快慢指针每次都移动一步，快指针仍然在环里
        // 当快慢指针相遇时，慢指针在环外，快指针在环内
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        // 求两个正序数组的中位数，相当于求第k大的元素
        int n = nums1.length;
        int m = nums2.length;
        if ((m + n) % 2 == 0) {
            // 如果是偶数，就要求第(m+n)/2和第(m+n)/2+1的平均
            return (findKth(nums1, 0, nums2, 0, (m + n) / 2) + findKth(nums1, 0, nums2, 0, (m + n) / 2 + 1)) / 2.0;
        } else {
            // 如果是奇数，求第(m+n)/2+1的值
            return findKth(nums1, 0, nums2, 0, (m + n) / 2 + 1);
        }
    }

    private static double findKth(int[] nums1, int s1, int[] nums2, int s2, int k) {
        // 求第k大的元素，每次都先比较第k/2大的元素，然后移除最小的那个k/2之前的元素

        // 确保第一个数组比第二个数组短
        if (nums1.length - s1 > nums2.length - s2) {
            return findKth(nums2, s2, nums1, s1, k);
        }

        // 如果第一个数组为空，直接返回第二个数组的第k个元素
        if (nums1.length == s1) return nums2[s2 + k - 1];

        // k=1，取两个数组中的最小值
        if (k == 1) {
            return Math.min(nums1[s1], nums2[s2]);
        }

        // num1继续取取k/2有可能越界
        int idx1 = Math.min(nums1.length, s1 + k / 2);
        // s2+2K,k-k/2是为了防溢出
        int idx2 = s2 + k - k / 2;
        // 如果num1的k/2小于num2的k/2，则把num1的k/2之前的元素都去掉，反之，去掉num2的k/2之前的元素
        if (nums1[idx1 - 1] < nums2[idx2 - 1]) {
            // 原本是取第k大的数，去掉k/2之前的元素后，就取第k -（去掉元素个数）大的元素。
            return findKth(nums1, idx1, nums2, s2, k - (idx1 - s1));
        } else {
            return findKth(nums1, s1, nums2, idx2, k - (idx2 - s2));
        }
    }

    public static int kthSmallest(TreeNode root, int k) {
        // 中序遍历，求二叉搜索树中第K小的元素
        List<Integer> list = new ArrayList<>();
        inorderTraversal(root, list, k);
        return list.get(k - 1);
    }

    private static void inorderTraversal(TreeNode node, List<Integer> list, int k) {
        if (node == null) {
            return;
        }
        if (list.size() > k) {
            return;
        }
        inorderTraversal(node.left, list, k);
        list.add(node.val);
        inorderTraversal(node.right, list, k);
    }

    static int maxPathSum = Integer.MIN_VALUE;

    public static int maxPathSum(TreeNode root) {
        maxPathSumDfs(root);
        return maxPathSum;
    }

    public static int maxPathSumDfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = Math.max(0, maxPathSumDfs(root.left));
        int right = Math.max(0, maxPathSumDfs(root.right));
        maxPathSum = Math.max(maxPathSum, left + right + root.val);
        return Math.max(left, right) + root.val;
    }

    public static ListNode mergeKLists(ListNode[] lists) {
        // 递归，把k个链表分成两半，
        int n = lists.length;
        if (n == 0) {
            return null;
        }
        if (n == 1) {
            return lists[0];
        }
        ListNode[] one = Arrays.copyOfRange(lists, 0, n / 2);
        ListNode[] two = Arrays.copyOfRange(lists, n / 2, n);
        ListNode listNode1 = mergeKLists(one);
        ListNode listNode2 = mergeKLists(two);
        return mergeTwoLists(listNode1, listNode2);
    }

    public static ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        // 递归，合并两个链表
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val < list2.val) {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
    }

    public static void sortColors(int[] nums) {
        int n0 = 0, n1 = 0;
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            nums[i] = 2;
            if (num == 1) {
                nums[n1] = 1;
                n1++;
            }
            if (num == 0) {
                nums[n1] = 1;
                nums[n0] = 0;
                n0++;
                n1++;
            }
        }
    }

    public static int majorityElement(int[] nums) {
        int vote = 0;
        int candidate = 0;
        for (int i = 0; i < nums.length; i++) {
            if (vote == 0) {
                candidate = nums[i];
            }
            vote += (nums[i] == candidate ? 1 : -1);
        }
        return candidate;
    }

    public static boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public static boolean isValidBST(TreeNode root, long min, long max) {
        if (root == null) {
            return true;
        }
        if (root.val >= max || root.val <= min) {
            return false;
        }
        return isValidBST(root.left, min, root.val) && isValidBST(root.right, root.val, max);
    }

    private static int widthOfBinaryTree = 0;

    public static int diameterOfBinaryTree(TreeNode root) {
        // 求二叉树的直径，相当于求一个二叉树的左子树深度 + 右子树深度
        diameterOfBinaryTreeDfs(root);
        return widthOfBinaryTree;
    }

    // 递归求当前二叉树的深度
    private static int diameterOfBinaryTreeDfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = diameterOfBinaryTreeDfs(root.left);
        int right = diameterOfBinaryTreeDfs(root.right);
        widthOfBinaryTree = Math.max(widthOfBinaryTree, left + right);
        return Math.max(left, right) + 1;
    }

    public static String longestWord(String[] words) {
        // 由于最长的字符都是他单词逐步添加一个字母组成，
        // 所以只要找到一个所有字符的结束位都是true，按字典顺序返回就可用了
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        String longest = "";
        for (String word : words) {
            if (trie.searchSerial(word)) {
                if (word.length() > longest.length() || (word.length() == longest.length() && word.compareTo(longest) < 0)) {
                    longest = word;
                }
            }
        }
        return longest;
    }

    public static boolean canFinish(int numCourses, int[][] prerequisites, int dfs) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        for (int i = 0; i < prerequisites.length; i++) {
            graph.get(prerequisites[i][1]).add(prerequisites[i][0]);
        }

        int[] visited = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            if (findCycle(i, visited, graph)) {
                return false;
            }
        }
        return true;
    }

    public static boolean findCycle(int node, int[] visited, List<List<Integer>> graph) {
        if (visited[node] == 1) {
            return true;
        }
        if (visited[node] == 2) {
            return false;
        }
        visited[node] = 1;
        for (Integer next : graph.get(node)) {
            if (findCycle(next, visited, graph)) {
                return true;
            }
        }
        // 标记为2，表示可通，没有循环
        visited[node] = 2;
        return false;
    }

    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        // 初始化图，key是课程，value是所有出度
        Map<Integer, List<Integer>> map = new HashMap<>();
        // 初始化所有课程的入度
        int[] indegree = new int[numCourses];
        for (int i = 0; i < prerequisites.length; i++) {
            indegree[prerequisites[i][0]]++;
            if (map.containsKey(prerequisites[i][1])) {
                map.get(prerequisites[i][1]).add(prerequisites[i][0]);
            } else {
                List<Integer> tmpList = new ArrayList<>();
                tmpList.add(prerequisites[i][0]);
                map.put(prerequisites[i][1], tmpList);
            }
        }

        // 通过bfs进行拓扑排序，把入度为0的课程入队
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegree.length; i++) {
            if (indegree[i] == 0) {
                queue.add(i);
            }
        }

        // 有环图
        if (queue.isEmpty()) {
            return false;
        }

        // bfs移除入度为0的课程
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Integer poll = queue.poll();
                // 取出该课程的所有出度,
                List<Integer> integers = map.get(poll);
                for (int i1 = 0; integers != null && i1 < integers.size(); i1++) {
                    // 移除入度为0的课程，同时指向的出度课程的入度-1
                    indegree[integers.get(i1)]--;

                    // 重复取入度为0的课程入队
                    if (indegree[integers.get(i1)] == 0) {
                        queue.add(integers.get(i1));
                    }
                }

            }
        }

        // 最后所有入度都为0，则没有环，该课程合理
        for (int i = 0; i < indegree.length; i++) {
            if (indegree[i] != 0) {
                return false;
            }
        }

        return true;
    }

    public static int orangesRotting(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        int flash = 0;
        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};

        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 腐烂的橘子入队
                if (grid[i][j] == 2) {
                    queue.add(new int[]{i, j});
                }
                // 统计新鲜橘子个数
                if (grid[i][j] == 1) {
                    flash++;
                }
            }
        }

        while (flash > 0 && !queue.isEmpty()) {
            res++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] poll = queue.poll();
                int row = poll[0];
                int col = poll[1];

                // 循环4个方向
                for (int j = 0; j < 4; j++) {
                    int newRow = row + dx[j];
                    int newCol = col + dy[j];

                    // 判断是否越界
                    if (newRow < 0 || newRow >= m || newCol < 0 || newCol >= n) {
                        continue;
                    }

                    // 把新鲜的橘子标记为腐烂
                    if (grid[newRow][newCol] == 1) {
                        grid[newRow][newCol] = 2;
                        flash--;
                        queue.offer(new int[]{newRow, newCol});
                    }
                }
            }
        }

        // 如果还有新鲜橘子，返回-1
        if (flash > 0) {
            return -1;
        }

        return res;
    }

    public static int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int nr = grid.length;
        int nc = grid[0].length;
        int res = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++res;
                    numIslandsDfs(grid, r, c);
                }
            }
        }

        return res;
    }

    private static void numIslandsDfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return;
        }

        grid[r][c] = '0';
        numIslandsDfs(grid, r - 1, c);
        numIslandsDfs(grid, r + 1, c);
        numIslandsDfs(grid, r, c - 1);
        numIslandsDfs(grid, r, c + 1);
    }

    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        // 通过双指针，倒序，把最大的那个原始放在最右边
        // num1指针
        int flag1 = m - 1;
        // num2指针
        int flag2 = n - 1;
        // 存放数据指针
        int flag3 = m + n - 1;

        while (flag1 >= 0 && flag2 >= 0) {
            if (nums1[flag1] > nums2[flag2]) {
                nums1[flag3] = nums1[flag1];
                flag3--;
                flag1--;
            } else {
                nums1[flag3] = nums2[flag2];
                flag3--;
                flag2--;
            }
        }

        while (flag2 >= 0) {
            nums1[flag3] = nums2[flag2];
            flag3--;
            flag2--;
        }
    }

    public static int findMin(int[] nums) {
        int start = 0;
        int end = nums.length - 1;
        while (start < end) {
            int mid = (end - start) / 2 + start;
            if (nums[mid] > nums[end]) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        return nums[start];
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
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            System.out.println(matrix[x][y]);
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            } else {
                ++x;
            }
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
