# 算法技巧类（43题）

## 1、暴力搜索-回溯

[46. 全排列（中等）](https://leetcode-cn.com/problems/permutations) 用一个list记录track，递归所有可能的情况，注意每次在track中放入一种情况并进入递归后，要把该情况移除，已达到“回溯”的效果

[51. N皇后（困难）](https://leetcode-cn.com/problems/n-queens) 回溯，只需判断上方、左上方和右上方三种情况，注意字符转换等细节即可

[698. 划分为k个相等的子集（中等）](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/) 以桶为视角进行装载和回溯。首先排除k>nums.length和sum%k!=0的情况，然后由sum/k得到target，构造一个used数组用来记录nums中对应位置的元素有没有被装入桶中，构造回溯函数backtrack(int k, int bucket, int[] nums, int start, boolean[] used, int target)，对于k=0的情况直接返回true，bucket=target的情况可以继续调用backtrack(k - 1, 0, nums, 0, used, target)，从start位置开始遍历nums，对于已被用过或装入后超出target的元素跳过，其他元素装入后如果backtrack(k, bucket, nums, i + 1, used, target)为true的话，可以直接返回true，其他情况将桶里的元素再拿出去。

[78. 子集（中等）](https://leetcode-cn.com/problems/subsets) 使用回溯法，用一个全局变量res记录结果，一个tmp记录某一子集的情况，backtrack(int[] nums, int cur)，对于每个nums[cur]，有取用和不取用两种状态，将nums[cur]塞入tmp，调用backtrack(nums, cur + 1)代表取用，将nums[cur]移出tmp并调用backtrack(nums, cur + 1)代表不取用，base case是cur >= nums.length的时候，将tmp塞入res

[77. 组合（中等）](https://leetcode-cn.com/problems/combinations) 思路同上一题，注意调整base case，tmp的size等于k时，塞入res，以及tmp的size加上n-cur+1如果小于k的话，说明剩下的元素不足以组成长度为k的组合了，直接返回

[37. 解数独（困难）](https://leetcode-cn.com/problems/sudoku-solver) 构建acktrack(char[][] board, int i, int j)函数，base case有三种：1、j=9则切换到下一行backtrack(board, i + 1, 0)；2、i=9则返回true；3、board[i][j] != '.'则继续下一个backtrack(board, i, j + 1)。然后处理回溯，char从1到9for循环，先校验字符合法性，然后在board[i][j]塞入ch，判断backtrack(board, i, j + 1)是否返回true，如果true则此处可以直接返回true，无需继续后面的循环，如果不是，则将board[i][j]重新置为'.'。校验字符合法性单独构造一个函数isValid(char[][] board, int i, int j, char ch)，对同行、同列、同九宫格的内容进行校验，注意同九宫格的校验公式

[22. 括号生成（中等）](https://leetcode-cn.com/problems/generate-parentheses) 只要控制两点即可：1.合法的组合左括号数量一定等于右括号数量；2.在成为合法组合前，左括号数量一定大于等于右括号数量。构建backtrack(int left, int right, StringBuilder s)函数，对于不合法的情况直接返回，对于合法的情况，将s塞入res然后返回，加左括号backtrack(left - 1, right, s)撤销，加右括号backtrack(left, right - 1, s)撤销

## 2、暴力搜索-BFS

[111. 二叉树的最小深度（简单）](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree) 类似层序遍历，用一个que存储每一层的节点，双重循环，外层!que.isEmpty()并depth++，内层用一个变量记录que.size，循环0~size，poll出一个节点，将左右子树offer进que

[752. 打开转盘锁（中等）](https://leetcode-cn.com/problems/open-the-lock) 每一位密码可以向上或向下扭动，有四位密码，所以每一次变动有2*4=8种可能的结果。用一个HashSet记录deadlist，使用双向BFS方法寻找答案，一个HashSet q1从“顶”开始（在题目中即为“000”），一个HashSet q2从“底”开始（在题目中即为目标值target），step初始化为0，双重循环，外循环新建一个HashSet tmp用来暂存当前节点的“下一层”，最后step++，q1=q2，q2=tmp。内循环遍历q1，在deadlist里则continue，在q2里则返回step（双向BFS是判断两个集合是否有交集），循环[0,4)，向上扭动，不在visited里则添加到tmp里，向下扭动，不在visited里则添加到tmp里

[773. 滑动谜题（困难）](https://leetcode-cn.com/problems/sliding-puzzle) 将题目中2*3的矩阵转化成一维，用String类型存储，方便比较，再用一个int[][] map记录String[i]可以向那几个位置移动，然后套BFS模板，每到达一个“节点”，调用get方法（写到主方法外面）获取下一层可能的情况List<String>，循环这个list

## 3、数学运算

[191. 位1的个数（简单）](https://leetcode-cn.com/problems/number-of-1-bits) 位运算是针对二进制的，&与运算，两个1结果为1，其他都为0（或者说只要有一个是0结果就是0），所以n&(n-1)可以消除n二进制表达里的最后一个1，所以只要对n不断进行该操作并计数，直到n被消除为0为止即可

[231. 2的幂（简单）](https://leetcode-cn.com/problems/power-of-two/) 一个数如果是2的指数，那么它的二进制表示一定只含有一个1，所以使用n&(n-1)判断即可

[136. 只出现一次的数字（简单）](https://leetcode-cn.com/problems/single-number/) 对于二进制而言，进行异或^操作，相同为0，不同为1。所以对十进制数字而言，两个相同的数异或^结果为0，一个数和0做异或结果还是它自身，两个非零且不同的数做异或结果要按二进制推导。所以对这题而言，只需要用初始化一个=0的res，再循环nums，使res ^= n，最后返回res即可。

[172. 阶乘后的零（中等）](https://leetcode-cn.com/problems/factorial-trailing-zeroes) 将问题转化为n!最多可以分解出多少个因子5。因为2*5=10会贡献0，因子2的个数又一定比5多，所以统计5的即可。d初始=n，循环res += d/5，d每次/5直到d/5<=0

[793. 阶乘后 K 个零（困难）](https://leetcode-cn.com/problems/preimage-size-of-factorial-zeroes-function) 搜索有多少个 n 满足f(n) == K，将问题转化为满足条件的 n 最小是多少，最大是多少，一减即可得出答案。使用二分搜索，找到f(n)=k时n的左边界和右边界。

[204. 计数质数（中等）](https://leetcode-cn.com/problems/count-primes) 如果一个数如果只能被 1 和它本身整除，那么这个数就是质数（素数）。初始化一个长度为n的数组isPrime将值全部填写为true，然后双重循环，外层从i开始到 i * i < n，i++，如果isPrime[i]为true，循环j=i * i到j < n，j += i，将isPrime[j]置为false，最后返回数组中true的数量即可

[372. 超级次方（中等）](https://leetcode-cn.com/problems/super-pow) 

[268. 丢失的数字（简单）](https://leetcode-cn.com/problems/missing-number/)

[645. 错误的集合（简单）](https://leetcode-cn.com/problems/set-mismatch)

[382. 链表随机节点（中等）](https://leetcode-cn.com/problems/linked-list-random-node)

[398. 随机数索引（中等）](https://leetcode-cn.com/problems/random-pick-index)

[292. Nim游戏（简单）](https://leetcode-cn.com/problems/nim-game) 如果对手拿的时候只剩 4 颗石子，那么无论他怎么拿，总会剩下 1~3 颗石子，就能赢。所以只要初始数量不是4的倍数（你就可以拿1~3颗给对手留下4的倍数），你就能赢。所以return n % 4 != 0;即可

[877. 石子游戏（中等）](https://leetcode-cn.com/problems/stone-game) 题目有两个条件很重要：1、石头总共有偶数堆，2、石头的总数是奇数。作为第一个拿石头的人，你可以控制自己拿到所有偶数堆，或者所有的奇数堆。你可以在第一步就观察好，奇数堆的石头总数多，还是偶数堆的石头总数多，然后步步为营，就一切尽在掌控之中了。所以先手必赢，直接return true即可

[319. 灯泡开关（中等）](https://leetcode-cn.com/problems/bulb-switcher) 假设现在总共有 16 盏灯，我们求 16 的平方根，等于 4，这就说明最后会有 4 盏灯亮着，它们分别是第 1*1=1 盏、第 2*2=4 盏、第 3*3=9 盏和第 4*4=16 盏。return (int)Math.sqrt(n);即可

## 4、其他技巧

[215. 数组中的第 K 个最大元素（中等）](https://leetcode-cn.com/problems/kth-largest-element-in-an-array)

[241. 为运算表达式设计优先级（中等）](https://leetcode-cn.com/problems/different-ways-to-add-parentheses)

[1288. 删除被覆盖区间（中等）](https://leetcode-cn.com/problems/remove-covered-intervals)

[56. 区间合并（中等）](https://leetcode-cn.com/problems/merge-intervals)

[986. 区间列表的交集（中等）](https://leetcode-cn.com/problems/interval-list-intersections)

## 5、经典面试题

[659. 分割数组为连续子序列（中等）](https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/)

[牛客网 吃葡萄](https://www.nowcoder.com/questionTerminal/14c0359fb77a48319f0122ec175c9ada)

[969. 煎饼排序（中等）](https://leetcode-cn.com/problems/pancake-sorting)

[43. 字符串相乘（中等）](https://leetcode-cn.com/problems/multiply-strings)

[224. 基本计算器（困难）](https://leetcode-cn.com/problems/basic-calculator)

[227. 基本计算器II（中等）](https://leetcode-cn.com/problems/basic-calculator-ii)

[772. 基本计算器III（困难）](https://leetcode-cn.com/problems/basic-calculator-iii)

[42. 接雨水（困难）](https://leetcode-cn.com/problems/trapping-rain-water)

[20. 有效的括号（简单）](https://leetcode-cn.com/problems/valid-parentheses)

[921. 使括号有效的最小添加（中等）](https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid)

[1541. 平衡括号串的最少插入（中等）](https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string)

[391. 完美矩形（困难）](https://leetcode-cn.com/problems/perfect-rectangle/)

[855. 考场就座（中等）](https://leetcode-cn.com/problems/exam-room)

[392. 判断子序列（简单）](https://leetcode-cn.com/problems/is-subsequence)