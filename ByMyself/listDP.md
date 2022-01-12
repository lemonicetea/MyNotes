# 做过的动态规划题分类（41题）

## 1、基本技巧

[509. 斐波那契数（简单）](https://leetcode-cn.com/problems/fibonacci-number) 1、使用给好的f(n)函数暴力解；2、利用备忘录+递归自顶向下解；3、使用DP table+迭代自底向上解

[322. 零钱兑换（中等）](https://leetcode-cn.com/problems/coin-change) 使用dp table，双重for循环，dp[i] = Math.min(dp[i], dp[i - coin] + 1)

[931. 下降路径最小和（中等）](https://leetcode-cn.com/problems/minimum-falling-path-sum/) memo+递归，注意base case和处理越界

[494. 目标和（中等）](https://leetcode-cn.com/problems/target-sum) 1、回溯法；2、带有备忘录的dp，key存字符串；3、使用sum[A]-sum[B]=target的公式推导出sum[B]=(sum[nums]-target)/2，从而将问题转化为：nums中可以凑出几个sum[B]子集，此问题就是经典的背包问题。

## 2、子序列类型问题

[72. 编辑距离（困难）](https://leetcode-cn.com/problems/edit-distance) 1、memo+递归；2、dp table+迭代。注意base case和处理越界

[354. 俄罗斯套娃信封问题（困难）](https://leetcode-cn.com/problems/russian-doll-envelopes) 将原数组重新排序，按“长”升序排序，长相等的再按“宽”降序排序，将问题转变为选取“宽”的最长递增子序列问题

[300. 最长递增子序列（中等）](https://leetcode-cn.com/problems/longest-increasing-subsequence) LIS（Longest Increasing Subsequence）1、使用dp table，双重循环，转移方程dp[i] = Math.max(dp[j] + 1, dp[i]) 2、二分查找+patience sorting（耐心排序），一个变量记录堆数，一个数组记录每堆对顶元素值用于比大小，最后返回堆数即可

[53. 最大子序和（简单）](https://leetcode-cn.com/problems/maximum-subarray/) 因为数组中可能包含复数，所以不能使用滑动窗口解决。动规dp[i]定义为“以nums[i]结尾的连续子数组最大和”，则dp[i] = Math.max(nums[i], nums[i] + dp[i - 1])

[1143. 最长公共子序列（中等）](https://leetcode-cn.com/problems/longest-common-subsequence) LCS（Longest Common Subsequence）1、递归，dp(s1, i, s2, j)计算s1[i..]和s2[j..]的最长公共子序列长度，从后向前；2、迭代，dp[i][j]代表s1[0..i-1]和s2[0..j-1]的最长公共子序列长度，base case为dp[0][..] = dp[..][0] = 0

[583. 两个字符串的删除操作（中等）](https://leetcode-cn.com/problems/delete-operation-for-two-strings/) 思路同LCS，res = n + m - 2 * dp[n][m]

[712. 两个字符串的最小ASCII删除和（中等）](https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings) 思路同LCS，res = sumASCII(s1+s2) - 2 * dp[n][m]，dp中存储公共序列的ASCII和

[516. 最长回文子序列（中等）](https://leetcode-cn.com/problems/longest-palindromic-subsequence) dp[i][j]表示子串s[i..j]内的最长回文子序列长度，最后返回dp[0][n - 1]即可，因为依赖dp[i+1][j-1]，所以要倒着或斜着填入数据

## 3、背包类型问题

[416. 分割等和子集（中等）](https://leetcode-cn.com/problems/partition-equal-subset-sum) 将问题转化为n个物品中是否可以恰好装满sum/2容量的包，1、dp[i][j]代表前i个物品当前容量为j时是否为true，最后返回dp[n][sum/2]，base case 就是dp[..][0] = true和dp[0][..] = false；2、将二维dp压缩成一维，因为只与j相关（压缩还需要多看，理解不透彻）

[518. 零钱兑换II（中等）](https://leetcode-cn.com/problems/coin-change-2) 1、dp[i][j]代表若只使用coins中的前i个硬币的面值，若想凑出金额j，有dp[i][j]种凑法，最后返回dp[n][amount]即可，注意结果是可能性和，所以装或不装两种情况的结果要相加；2、压缩空间，降维，只与j相关

## 4、游戏专题

[64. 最小路径和（中等）](https://leetcode-cn.com/problems/minimum-path-sum) base case是第一行和第一列，其余dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1])+nums[i][j]

[174. 地下城游戏（困难）](https://leetcode-cn.com/problems/dungeon-game) 推荐使用带备忘录的递归解法，比较容易理解。memo[i][j]的含义是从dungeon[i][j]到达终点（右下角）所需的最少生命值是dp(dungeon, i, j)，那么就需要判断Math.min(dp(dungeon, i + 1, j), dp(dungeon, i, j + 1)) - dungeon[i][j]是否大于0，如果大于0则沿用当前值，如果小于等于0则说明dungeon[i][j]会补充足够的能量，保持最低生命1即可，最后返回memo[0][0]

[514. 自由之路（困难）](https://leetcode-cn.com/problems/freedom-trail/) 带备忘录的递归解法，需要将ring字符串处理成HashMap<Character, ArrayList<Integer>>的表，以方便找到目标字符与当前所在字符所需走过的路径值。dp函数的含义为当圆盘指针指向 ring[i] 时，输入字符串 key[j..] 至少需要 dp(ring, i, key, j) 次操作

[787. K 站中转内最便宜的航班（中等）](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/)

[10. 正则表达式匹配（困难）](https://leetcode-cn.com/problems/regular-expression-matching/) 用dp(s,i,p,j)表示s[i..]是否可以匹配p[j..]然后递归得到答案，转移方程分为：1.1 通配符匹配 0 次或多次res = dp(s, i + 1, p, j) || dp(s, i, p, j + 2)；1.2 常规匹配 1 次res = dp(s, i + 1, p, j + 1)；2.1 通配符匹配 0 次res = dp(s, i, p, j + 2)；2.2 如果没有*通配符，也无法匹配，那只能说明匹配失败。base case分别处理i == sl和就== pl的情况。使用HashMap<String, Boolean> memo来记录已经计算过的内容。

[887. 鸡蛋掉落（困难）](https://leetcode-cn.com/problems/super-egg-drop/) 有以下解法：1、易于理解，部分用例会超时。dp(k, n)表示k个鸡蛋n层楼尝试的最少次数，转移方程为res = Math.min(res, 1 + Math.max(dp(k - 1, i - 1), dp(k, n - i)))，取max是为了考虑最坏情况，base case为k = 1返回n，n = 0返回0，哈希备忘录memo(k+n, res)；2、将1中for循环的部分改为二分搜索，转化为求山谷问题；3、转变思路，用dp[K][m] = n来表示给你 K 个鸡蛋，测试 m 次，最坏情况下最多能测试 n 层楼，最后找到题目要求的n对应的m即可，转移方程为dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1；4、将3中m循环改为二分（没做出来）

[312. 戳气球（困难）](https://leetcode-cn.com/problems/burst-balloons) 构造points[n+2]，一头一尾存1，转移方程dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[k][j] + points[i] * points[k] * points[j]) dp取开区间i到j内可得到的最大数，所以最终返回dp[0][n + 1]即可，base case为所有i=j时，dp取值为0。注意确定base case和最终要得到的状态后，画出dp矩阵，即可推出遍历方向。

[877. 石子游戏（中等）](https://leetcode-cn.com/problems/stone-game) 1、先手必胜，一定返回true；2、follow up 堆数和总数都是任意正整数，用dp[i][j]存储piles[i...j]这部分石头堆先后手各能获得的最高分数

[651. 四键键盘（中等） Plus会员](https://leetcode-cn.com/problems/4-keys-keyboard) 1、定义dp(n, a_num, copy)函数，用n表示剩余的按键次数，用a_num表示当前屏幕上A的数量，用copy表示当前剪切板中A的数量，每次对# A、# C-V、# C-A C-C这三种状态取最大，直到n减为0，用备忘录减少子集；2、定义dp[n + 1]，用dp[i]表示i次操作后最多能显示多少个A，转移方程dp[i]=Math.max(dp[i], dp[j-2]*(i-j+1))，j之前经历了C-A和C-C，j到i之间全部C-V

[121. 买卖股票的最佳时机（简单）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) 套模板，k = 1，不需要加上前一次的利润

[122. 买卖股票的最佳时机 II（中等）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/) 套模板，k无限制，需要加上前一次的利润

[123. 买卖股票的最佳时机 III（困难）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/) 套模板，k = 2，因为情况较少，所以可以直接穷举，一种是k=1（状态转移方程2个，不需要加上前一次的利润），一种是k=2（状态方程2个，需要加上前一次的利润）

[188. 买卖股票的最佳时机 IV（困难）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/) 套模板，k有比较大的上限，需要三维dp table，最复杂的一种情况，套模板写全细节，最后返回dp[n - 1][max_k][0]即可，但是效率会很低

[309. 最佳买卖股票时机含冷冻期（中等）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) 套模板，k无限制，有冻结期，需要加上前一次的利润，时间由i - 1变为i - (冻结期 + 1)

[714. 买卖股票的最佳时机含手续费（中等）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) 套模板，k无限制，有手续费，需要加上前一次的利润，购买股票时除了减去prices[i]还要减去fee

[198. 打家劫舍（简单）](https://leetcode-cn.com/problems/house-robber) 倒序，dp[i] = Math.max(dp[i + 1], dp[i + 2] + nums[i])

[213. 打家劫舍II（中等）](https://leetcode-cn.com/problems/house-robber-ii) 写一个辅助函数dp(int[] nums, int start, int end)，求Math.max(dp(nums, 0, n - 2), dp(nums, 1, n - 1))即可

[337. 打家劫舍III（中等）](https://leetcode-cn.com/problems/house-robber-iii) 写一个辅助函数dp(TreeNode root)，返回结果为int[]{not_rob, rob}，该节点偷则为root.val + left[0] + right[0]，不偷则为Math.max(left[0], left[1]) + Math.max(right[0], right[1])

[28. 实现 strStr(简单)](https://leetcode-cn.com/problems/implement-strstr) 扩展 KMP算法（Knuth-Morris-Pratt 算法）

[1312. 让字符串成为回文串的最少插入次数（困难）](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome) dp[n][n]，i==j时为0（base case） 双重for循环倒序遍历dp数组，s[i] == s[j]时dp[i][j] = dp[i+1][j-1]，不等时dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1

## 5、贪心类型问题

[435. 无重叠区间（中等）](https://leetcode-cn.com/problems/non-overlapping-intervals/) 将数组按区间end时间升序排序，找到下一个start >= x_end的区间，count++，更新end，直到遍历完整个数组

[452. 用最少数量的箭引爆气球（中等）](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons) 思路同上一题，将>=改为>即可

[252.会议室（简单） Plus会员](https://leetcode-cn.com/problems/meeting-rooms/) 同II

[253.会议室 II（中等） Plus会员](https://leetcode.com/problems/meeting-rooms-ii/) 将所有区间的开始和结束时间分别投影到int[] begin和int[] end数组中，分别对begin和end进行升序排序，用一个变量count记录扫描过程中的得到的所需会议室的总值，i和j作为双指针分别扫描begin和end，若begin[i] < end[j]说明扫描到一个开始点，count++，i++，否则count--，j++，然后判断res = Math.max(res, count)，最终返回res

[1024. 视频拼接（中等）](https://leetcode-cn.com/problems/video-stitching) 将clips按照起点升序排序，起点相同的按照终点降序排序，定义curEnd和nextEnd两个变量，遍历clips中所有起点<=curEnd的clip，不断刷新nextEnd的值，遍历完后将nextEnd的值赋给curEnd，同时res++，若curEnd已经>=目标T，则可以给出答案，否则继续遍历clips直到结束，返回-1

[55. 跳跃游戏（中等）](https://leetcode-cn.com/problems/jump-game) 判断是否能到终点。用一个变量farthest记录能到达的最远index，for循环不断计算能跳到的最远距离farthest = max(farthest, i + nums[i])，如果遇到farthest <= i则说明无法前进了，直接返回false，最后判断farthest >= n - 1即可

[45. 跳跃游戏 II（中等）](https://leetcode-cn.com/problems/jump-game-ii) 判断到达终点的最少跳跃次数。用一个变量jumps记录需要跳跃的最少次数，end记录当前到达的index，farthest记录当前可到达的最远index，farthest = max(nums[i] + i, farthest)，如果end恰好是当前循环到的i，说明可以进行下一次跳跃了，jumps++，end更新为farthest，循环结束后返回jumps即可

[134. 加油站（中等）](https://leetcode-cn.com/problems/gas-station/) 1、图像解法。sum += gas[i] - cost[i]计算走完所有节点后剩余的油量，每次都比较一下sum和minSum，若小于，start = i + 1，minsum更新为sum。最后如果sum<0，说明总油量小于总消耗，-1。最后返回start == n ? 0 : start，处理一下环的情况；2、贪心解法。依然计算sum，小于0返回-1。用一个变量tank记录当前油箱里的油量，tank += gas[i] - cost[i]，如果tank<0，说明无法从start走到i，将start更新为i+1，tank更新为0，继续循环。最后判断start == n ? 0 : start