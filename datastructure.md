# 一、栈 Stack

## 1.括号题

```java
// 最长有效括号
class Solution {
    public int longestValidParentheses(String s) {
        int res = 0, start = 0;
        Stack<Integer> leftIdx = new Stack<>();
        for(int i = 0; i < s.length(); ++i) {
            if(s.charAt(i) == '(')
                leftIdx.push(i);
            else {
                if(leftIdx.isEmpty()) start = i + 1;
                else {
                    leftIdx.pop();
                    if(leftIdx.isEmpty()) res = Math.max(res, i - start + 1);
                    else res = Math.max(res, i - leftIdx.peek());
                }                
            }
        }
        return res;
    }
}
```

## 2.基本计算器

```java
// 只有加减乘除和空格
class Solution {
    public int calculate(String s) {
        if(s == null || s.length() == 0)
            return 0;
        Stack<Character> operators = new Stack<>();
        Stack<Integer> nums = new Stack<>();
        for(int i = 0; i < s.length(); ) {
            while(i < s.length() && s.charAt(i) == ' ')
                ++i;
            if(i == s.length()) break;
            if(s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                int num = 0;
                while(i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                    num = num * 10 + (s.charAt(i++) - '0');
                }
                nums.push(num);
            } else {
                while(!operators.isEmpty() && getLevel(operators.peek()) >= getLevel(s.charAt(i))) {
                    calculateTwoNums(nums, operators);
                }
                operators.push(s.charAt(i++));
            }
        }
        while(!operators.isEmpty()) {
            calculateTwoNums(nums, operators);
        }
        return nums.peek();
    }

    private int getLevel(char ope) {
        return ope == '+' || ope == '-' ? 1 : 2;
    }

    private void calculateTwoNums(Stack<Integer> nums, Stack<Character> operators) {
        int num2 = nums.pop();
        int num1 = nums.pop();
        int temp = 0;
        switch(operators.pop()) {
            case '+':
                temp = num1 + num2;
                break;
            case '-':
                temp = num1 - num2;
                break;
            case '*':
                temp = num1 * num2;
                break;
            case '/':
                temp = num1 / num2;
        }
        nums.push(temp);
    }
}

// 只有加减和括号
class Solution {
    public int calculate(String s) {
        // 将所有括号打开，如果左括号前面是减号，则这对括号里面的减号都变成加号
        int res = 0;
        int sign = 1;
        int num = 0; // 考虑多位数
        Stack<Integer> signBeforeParen = new Stack<>(); // 记录左括号前面的符号，用于判断括号内的符号
        signBeforeParen.push(sign);
        for(int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if(c >= '0' && c <= '9')
                num = num * 10 + c - '0';
            else if(c == '+' || c == '-') {
                res += sign * num;
                sign = signBeforeParen.peek() * (c == '+' ? 1 : -1);
                num = 0;
            } else if(c == '(') signBeforeParen.push(sign);
            else if(c == ')') signBeforeParen.pop();
        }

        res += sign * num; // num可能不等于0
        return res;
    }
}
```

## 3.直方图题

```java
// 接雨水
class Solution {
    public int trap(int[] height) {
        if(height.length < 3) return 0;
        int maxArea = 0, n = height.length;
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < n - 1; ++i) {
            // 下坡入栈
            if(height[i] > height[i + 1]) {
                stack.push(i);
                continue;
            }
            // 平地不入栈
            if(height[i] == height[i + 1]) continue;
            // 遇到上坡，求困水面积
            int leftWall = i;
            while(!stack.isEmpty() && height[stack.peek()] <= height[i + 1])
                leftWall = stack.pop();
            if(!stack.isEmpty())
                leftWall = stack.peek(); // 左墙比右墙高，左墙不出栈
            int h = Math.min(height[leftWall], height[i + 1]);
            for(int j = i; j > leftWall; --j) {
                maxArea += h - height[j];
                height[j] = h;
            }
        }
        return maxArea;
    }
}

// 直方图中的最大矩形面积
class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> leftBorders = new Stack<>(); // 栈中存的都是递增的元素
        int ans = 0, p = 0;
        while(p < heights.length) {
            // 如果当前栈中为空 或 i元素比栈顶元素大，入栈
            if(leftBorders.isEmpty() || heights[p] >= heights[leftBorders.peek()]) {
                leftBorders.push(p++);
                continue;
            }
            // p元素比栈顶元素小时，可求出完全包裹栈顶bar的矩形面积
            // 栈顶出栈后，新的栈顶是左边界，p元素是右边界
            int cur = leftBorders.pop();
            int left = leftBorders.isEmpty() ? -1 : leftBorders.peek();
            ans = Math.max(ans, (p - left - 1) * heights[cur]);
        }
        
        // 遍历完如果栈不为空 说明栈中元素全是递增的
        while(!leftBorders.isEmpty()) {
            int cur = leftBorders.pop();
            // 左边界
            int left = leftBorders.isEmpty() ? -1 : leftBorders.peek();
            // 现在cur右边没有比它更小的元素，所以右边界设置为heights.size()
            ans = Math.max(ans, (p - left - 1) * heights[cur]);
        }
        
        return ans;
    }
}
```

## 4.下一个更大的元素

```java
// 每日温度
class Solution {
    public int[] dailyTemperatures(int[] T) {
        if(T == null) return null;
        int n = T.length;
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
                // 从右向左
        for(int i = n - 1; i >= 0; --i) {
            while(!stack.isEmpty() && T[i] >= T[stack.peek()])
                stack.pop();
            if(stack.isEmpty()) res[i] = 0;
            else res[i] = stack.peek() - i;
            stack.push(i);
        }
        return res;
    }
}

// 循环数组中下一个更大的元素
// 每个元素都入栈两次
public class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * nums.length - 1; i >= 0; --i) {
            while (!stack.empty() && nums[stack.peek()] <= nums[i % nums.length]) {
                stack.pop();
            }
            res[i % nums.length] = stack.empty() ? -1 : nums[stack.peek()];
            stack.push(i % nums.length);
        }
        return res;
    }
}

// 移除k位数字后最大的数
class Solution {
    public String removeKdigits(String num, int k) {
        int digitsNum = num.length() - k;
        if(digitsNum == 0) return "0";
        // 用数组模拟栈
        char[] stack = new char[num.length()];
        int topIdx = -1;
        // 递增数字依次入栈，直到出现比栈顶元素小的数字c
        // 将栈中所有比c小的数字出栈（删除这些数字）
        for(int i = 0; i < num.length(); ++i) {
            // 栈中比当前数字大的出栈
            char c = num.charAt(i);
            while(topIdx >= 0 && stack[topIdx] > c && k > 0) {
                --topIdx;
                --k;
            }
            // 此时栈中元素均比c小
            stack[++topIdx] = c;
        }
        int startIdx = 0;
        while(startIdx <= topIdx && stack[startIdx] == '0')
            ++startIdx;
        return startIdx > topIdx ? "0" : new String(stack, startIdx, digitsNum - startIdx);
    }
}
```

## 5.最小栈

```java
// 两个栈
class MinStack {
    
    Stack<Integer> nums;
    Stack<Integer> min;
    /** initialize your data structure here. */
    public MinStack() {
        nums = new Stack<>();
        min = new Stack<>();
    }
    
    public void push(int x) {
        nums.push(x);
        if(min.empty() || x <= min.peek())
            min.push(x);
    }
    
    public void pop() {
        int peek = nums.pop();
        if(peek == min.peek())
            min.pop();
    }
    
    public int top() {
        return nums.peek();
    }
    
    public int getMin() {
        return min.peek();
    }
}

// 一个栈
class MinStack {

    Deque<Integer> stack = null;
    int min = Integer.MAX_VALUE;
    // 题目重点在于getMin()，如何能够在元素出栈时o(1)维护min变量
    /** initialize your data structure here. */
    public MinStack() {
        stack = new LinkedList<>();
    }
    
    public void push(int x) {
        // 当入栈元素是最小元素时
        if(x <= this.min) {
            stack.push(min);
            min = x;
        }
        stack.push(x);
    }
    
    public void pop() {
        // 当栈顶元素是最小元素时
        if(stack.pop() == min)
            min = stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return min;
    }
}

// 一个栈：最省空间
class MinStack {

    // 第二种方法：更省空间
    // 栈内存储入栈元素与当前min的差值
    // 出栈的时候，如果出栈的是负数，说明出的就是min，同时恢复上一个min = min - stack.peek()
    Deque<Long> stack = null; // 因为要作差，所以栈内元素值会超过int的范围
    long min;
    /** initialize your data structure here. */
    public MinStack() {
        stack = new LinkedList<>();
    }
    
    public void push(int x) {
        if(stack.isEmpty())
            min = x;
        stack.push(x - min);
        if(x < min)
            min = x;
    }
    
    public void pop() {
        if(stack.peek() < 0)
            min = min - stack.peek();
        stack.pop();
    }
    
    public int top() {
        return stack.peek() < 0 ? (int) min : (int) (min + stack.peek());
    }
    
    public int getMin() {
        return (int) min;
    }
}
```

## 6.函数的独占时间

```java
class Solution {
    private String START = "start";
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>(); // 栈内存放的是未结束的函数id
        String[] content = logs.get(0).split(":");
        stack.push(Integer.parseInt(content[0])); // 第一条日志一定是start的
        int preTime = Integer.parseInt(content[2]);
        for(int i = 1; i < logs.size(); ++i) {
            content = logs.get(i).split(":");
            if(START.equals(content[1])) {
                if(!stack.isEmpty()) // 有新的函数被调用，累加一部分栈顶函数的独占时间
                    res[stack.peek()] += Integer.parseInt(content[2]) - preTime;
                stack.push(Integer.parseInt(content[0]));
                preTime = Integer.parseInt(content[2]);
            } else {
                // 栈顶函数调用完毕
                res[stack.pop()] += Integer.parseInt(content[2]) - preTime + 1;
                preTime = Integer.parseInt(content[2]) + 1;
            }
        }
        return res;
    }
}
```

# 二、队列 Queue，Deque

## 1.滑动窗口——双向队列

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[Math.max(0, nums.length - k) + 1];
        if(nums == null || nums.length == 0)
            return new int[] {};
        // 第一个窗
        for(int i = 0; i < k; ++i) {
            while(!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
                deque.pollLast();
            }
            deque.addLast(i);
        }
        res[0] = nums[deque.peekFirst()];
        for(int i = k; i < nums.length; ++i) {
            if(deque.peekFirst() <= i - k) deque.pollFirst();
            while(!deque.isEmpty() && nums[i] >= nums[deque.peekLast()])
                deque.pollLast();
            deque.addLast(i);
            res[i - k + 1] = nums[deque.peekFirst()];
        }
        return res;
    }
}
```

# 三、优先队列 PriorityQueue

## 1.第k小/大/高频的元素

```java
// 合并k个排序链表
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0)
            return null;
        PriorityQueue<ListNode> nodes = new PriorityQueue<>((a, b) -> a.val - b.val);
        for(ListNode head : lists)
            if(head != null)
                nodes.add(head);
        
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while(!nodes.isEmpty()) {
            cur.next = nodes.poll();
            cur = cur.next;
            if(cur.next != null)
                nodes.add(cur.next);
        }
        return dummy.next;
    }
}

// 未排序数组中第k大的元素
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for(int num : nums) {
            pq.offer(num);
            if(pq.size() > k)
                pq.remove();
        }
        return pq.peek();
    }
}

// 前k个高频单词
class Solution {
    public List<String> topKFrequent(String[] words, int k) {
        HashMap<String, Integer> map = new HashMap<>();
        for(String word : words)
            map.put(word, map.getOrDefault(word, 0) + 1);
        PriorityQueue<String> pq = new PriorityQueue<>((a, b) -> map.get(a) == map.get(b) ? b.compareTo(a) : map.get(a) - map.get(b));
        for(String word : map.keySet()) {
            pq.offer(word);
            if(pq.size() > k)
                pq.poll();
        }
        List<String> res = new LinkedList<>();
        while(!pq.isEmpty())
            res.add(0, pq.poll());
        return res;
    }
}
```

## 2.数据流的中位数

```java
class MedianFinder {

    PriorityQueue<Integer> left = null;
    PriorityQueue<Integer> right = null;
    /** initialize your data structure here. */
    public MedianFinder() {
        // 左半边数据用最大堆
        left = new PriorityQueue<>((a, b) -> b - a);
        // 右半边数据用最小堆
        right = new PriorityQueue<>((a, b) -> a - b);
    }
    
    public void addNum(int num) {
        int size = left.size() + right.size();
        if(size % 2 == 0) {
            // 第奇数个数字加入到最大堆（左边）
            if(left.isEmpty()) {
                // num是数据流中的第一个数字
                left.add(num);
                return;
            }
            if(num > right.peek()) {
                left.add(right.poll());
                right.add(num);
            } else {
                left.add(num);
            }
        } else {
            if(num < left.peek()) {
                right.add(left.poll());
                left.add(num);
            } else {
                right.add(num);
            }
        }
    }
    
    public double findMedian() {
        int size = left.size() + right.size();
        if(size % 2 == 0)
            return (left.peek() + right.peek()) * 1.0 / 2;
        return left.peek() * 1.0;
    }
}
```

## 3.排列问题

```java
// 重构字符串
// 尽量先排个数多的字母
class Solution {
    public String reorganizeString(String S) {
        if(S.length() <= 1) return S;
        int[] count = new int[26];
        for(int i = 0; i < S.length(); ++i)
            ++count[S.charAt(i) - 'a'];
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for(int i = 0; i < 26; ++i) {
            if(count[i] != 0) {
                int[] pair = new int[2];
                pair[0] = i;
                pair[1] = count[i];
                pq.offer(pair);
            }
        }

        int lastChar = -1;
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < S.length(); ++i) {
            if(pq.peek()[0] == lastChar) {
                int[] temp = pq.poll();
                if(pq.isEmpty()) return "";
                int[] peek = pq.poll();
                sb.append((char)(peek[0] + 'a'));
                if(--peek[1] > 0) pq.offer(peek);
                pq.offer(temp);
                lastChar = peek[0];
            } else {
                int[] peek = pq.poll();
                sb.append((char)(peek[0] + 'a'));
                if(--peek[1] > 0) pq.offer(peek);
                lastChar = peek[0];
            }
        }
        return sb.toString();
    }
}

// 任务调度器
class Solution {
    public int leastInterval(char[] tasks, int n) {
        if(tasks == null) return 0;

        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        int[] counter = new int[26];
        for(char task : tasks)
            ++counter[task - 'A'];
        int tasksCount = 0;
        for(int i = 0; i < 26; ++i) {
            if(counter[i] != 0) {
                pq.offer(counter[i]);
                tasksCount += counter[i];
            }
        }

        int time = 0;
        while(!pq.isEmpty()) {
            int tasksNum = pq.size(); // 任务种类数目
            List<Integer> temp = new ArrayList<>();
            for(int i = 0; i <= n; ++i) {
                if(!pq.isEmpty()) {
                    int task = pq.poll();
                    if(task > 1) {
                        --task;
                        temp.add(task);
                    }
                }
                ++time;
                if(pq.isEmpty() && temp.isEmpty()) break;
            }
            for(int task : temp)
                pq.offer(task);
        }

        return time;
    }
}

// 会议室II
class Solution {
    public int minMeetingRooms(int[][] intevals) {
        if(intervals.length == 0 || intervals[0].length == 0)
            return 0;

        Arrays.sort(intervals, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        PriorityQueue<Integer> rooms = new PriorityQueue<>();
        rooms.offer(intervals[0][1]);
        for(int i = 1; i < intervals.length; ++i) {
            if(intervals[i][0] >= rooms.peek())
                rooms.poll();
            rooms.offer(intervals[i][1]);
        }
        return rooms.size();
    }
}
```

# 四、红黑树

```java
// 存在重复元素III：两数绝对值之差小于等于k，索引之差小于等于t
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> window = new TreeSet<>();
        for(int i = 0; i < nums.length; ++i) {
            // 红黑树的查找、插入与删除操作的时间复杂度均为O(nlog(min(n, k)))
            long num = (long)nums[i];
            Long greaterMin = window.ceiling(num);
            if(greaterMin != null && greaterMin - num <= (long)t)
                return true;
            Long lessMax = window.floor(num);
            if(lessMax != null && num - lessMax <= (long)t)
                return true;
            window.add(num);
            if(window.size() > k)
                window.remove((long)nums[i - k]);
        }
        return false;
    }
}

// 滑动窗口最大值
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums == null) return null;
        int n = nums.length;
        int[] res = new int[n - k + 1];
        TreeMap<Integer, Integer> map = new TreeMap<>(Collections.reverseOrder());
        for(int i = 0; i < k; ++i)
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        res[0] = map.firstKey();
        for(int i = 1; i < res.length; ++i) {
            if(map.get(nums[i - 1]) == 1)
                map.remove(nums[i - 1]);
            else
                map.put(nums[i - 1], map.get(nums[i - 1]) - 1);
            map.put(nums[i + k - 1], map.getOrDefault(nums[i + k - 1], 0) + 1);
            res[i] = map.firstKey();
        }
        return res;
    }
}

// 天际线问题
class Solution {
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> res = new ArrayList<>();
        if(buildings.length == 0 || buildings[0].length == 0) return res;
        // 将建筑的左上点和右上点放进高度数组中
        List<int[]> heights = new ArrayList<>();
        for(int[] building : buildings) {
            heights.add(new int[] {building[0], -building[2]}); // 左墙设为负值
            heights.add(new int[] {building[1], building[2]}); // 右墙设为正值
        }
        // 将这些点按坐标从左到右、高度从大到小的方式排序
        heights.sort((a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        // 根据排好序的点，搜取拐点写入res
        // 按<height, count>统计最高点，遇到拐点
        TreeMap<Integer, Integer> heightMap = new TreeMap<>(Collections.reverseOrder());
        heightMap.put(0, 1); // 地平线
        int preHeight = 0, curHeight = 0;
        for(int[] h : heights) {
            if(h[1] < 0) { // 左上点
                int count = heightMap.getOrDefault(-h[1], 0);
                heightMap.put(-h[1], count + 1);
            } else { // 右上角
                int count = heightMap.get(h[1]);
                if(count == 1) heightMap.remove(h[1]);
                else heightMap.put(h[1], count - 1);
            }
            curHeight = heightMap.firstKey(); // 当前高度就是没有碰到右上角的所有楼的最高高度
            if(curHeight != preHeight) {
                // 当前高度与前一高度不相等，说明碰到拐点
                Integer[] point = new Integer[] {h[0], curHeight};
                res.add(Arrays.asList(point));
                preHeight = curHeight;
            }
        }
        return res;
    }
}
```