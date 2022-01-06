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

[10. 正则表达式匹配（困难）](https://leetcode-cn.com/problems/regular-expression-matching/)

[887. 鸡蛋掉落（困难）](https://leetcode-cn.com/problems/super-egg-drop/)

[312. 戳气球（困难）](https://leetcode-cn.com/problems/burst-balloons)

[877. 石子游戏（中等）](https://leetcode-cn.com/problems/stone-game)

[651. 四键键盘（中等）](https://leetcode-cn.com/problems/4-keys-keyboard)

[121. 买卖股票的最佳时机（简单）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

[122. 买卖股票的最佳时机 II（简单）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

[123. 买卖股票的最佳时机 III（困难）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

[188. 买卖股票的最佳时机 IV（困难）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

[309. 最佳买卖股票时机含冷冻期（中等）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

[714. 买卖股票的最佳时机含手续费（中等）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

[198. 打家劫舍（简单）](https://leetcode-cn.com/problems/house-robber)

[213. 打家劫舍II（中等）](https://leetcode-cn.com/problems/house-robber-ii)

[337. 打家劫舍III（中等）](https://leetcode-cn.com/problems/house-robber-iii)

[28. 实现 strStr(简单)](https://leetcode-cn.com/problems/implement-strstr)

[1312. 让字符串成为回文串的最少插入次数（困难）](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome)

## 5、贪心类型问题

[435. 无重叠区间（中等）](https://leetcode-cn.com/problems/non-overlapping-intervals/)

[452. 用最少数量的箭引爆气球（中等）](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons)

[253.会议室 II（中等）](https://leetcode.com/problems/meeting-rooms-ii/)

[1024. 视频拼接（中等）](https://leetcode-cn.com/problems/video-stitching)

[55. 跳跃游戏（中等）](https://leetcode-cn.com/problems/jump-game)

[45. 跳跃游戏 II（中等）](https://leetcode-cn.com/problems/jump-game-ii)

[134. 加油站（中等）](https://leetcode-cn.com/problems/gas-station/)