# 算法技巧类（44题）

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

[372. 超级次方（中等）](https://leetcode-cn.com/problems/super-pow) 将问题转化一下，首先解决次方数是数组的问题，发现a^[1,5,6,4] = a^4 * (a^[1,5,6])^10，可以递归解决。然后解决模除问题，结论是(a * b)%k = (a%k)(b%k)%k，推导过程：假设ABCD是常数，a=Ak+B；b=Ck+D => ab=ACk^2+ADk+BCk+BD => ab%k=BD%k，a%k=B；b%k=D => (a%k)(b%k)%k=BD%k => (a*b)%k=(a%k)(b%k)%k。所以递归的每个子问题都可以被拆解为如例：((a^4%k) * (a^[1,5,6])^10%k) % k，其中a^4%k=(a%k * a%k * a%k * a%k)%k=(a%k)(a^3%k)%k，所以除了递归函数以外，还可以再写一个求模的辅助函数getMod，a %= base，res从1开始，每次res *= a;res %= base;循环k次。优化：还可以用幂运算的规律进一步提高效率，对于a^k，如果k为奇数则a^k=a * a^k-1，如果k为偶数则a^k=(a^k/2)^2，那么就有机会将for循环的问题降成log级别。基于这个思路优化getMod方法，将其改造为递归函数，这就是快速幂算法

[268. 丢失的数字（简单）](https://leetcode-cn.com/problems/missing-number/) 方法一：利用异或运算的规律可以发现：2 ^ 3 ^ 2 = 3 ^ (2 ^ 2) = 3 ^ 0 = 3，这个问题有点类似136.只出现一次的数字。对于这题，我们只要把nums的所有数和[0, n]的数都来一遍，成对的就会被消除，留下落单（丢失）的那个数了。方法二：[0, n]求和减去sum(nums)即可得出答案，[0, n]求和可以用等差数列求和公式(0 + n) * (n + 1) / 2。方法三：等差数列求和时可能有整型溢出的风险，所以可以把方法一二做个结合，让每个索引减去其对应的元素，再把相减的结果加起来，补上n-0，就能算出丢失的元素

[645. 错误的集合（简单）](https://leetcode-cn.com/problems/set-mismatch) 丢失一个元素并且重复了一个元素。元素和索引是成对儿出现的，常用的方法是排序、异或、映射。我们通过对每个元素减一达到索引和元素映射起来，通过改变元素的正负符号记录这个元素是否被映射。所以解决方法是循环两遍，第一遍取nums[i]的绝对值减一得到正确索引，如果nums[index]小于零说明已经被映射过，得到重复值，否则将nums[index]的值转为负数。第二遍找到nums[i]大于零的i将其加一得到丢失的数

[382. 链表随机节点（中等）](https://leetcode-cn.com/problems/linked-list-random-node) 需要均匀随机的返回某个节点的val，每个节点被选中的概率是1/n，可以转化为，当遇到第i个元素时，应该有1/i的概率选择该元素，1 - 1/i的概率保持原有的选择（通过公式推导可以证明）。Random对象的nextInt(i)方法可以生成一个[0, i)之间的整数，每次以1/i的概率更新结果即可。扩展一下，如果要随机选择k个数，只要在第i个元素处以k/i的概率选择该元素，以1 - k/i的概率保持原有选择即可。

[398. 随机数索引（中等）](https://leetcode-cn.com/problems/random-pick-index) 可以在上一题的基础上多加一层==target的判断，其他逻辑一样

[292. Nim游戏（简单）](https://leetcode-cn.com/problems/nim-game) 如果对手拿的时候只剩 4 颗石子，那么无论他怎么拿，总会剩下 1~3 颗石子，就能赢。所以只要初始数量不是4的倍数（你就可以拿1~3颗给对手留下4的倍数），你就能赢。所以return n % 4 != 0;即可

[877. 石子游戏（中等）](https://leetcode-cn.com/problems/stone-game) 题目有两个条件很重要：1、石头总共有偶数堆，2、石头的总数是奇数。作为第一个拿石头的人，你可以控制自己拿到所有偶数堆，或者所有的奇数堆。你可以在第一步就观察好，奇数堆的石头总数多，还是偶数堆的石头总数多，然后步步为营，就一切尽在掌控之中了。所以先手必赢，直接return true即可

[319. 灯泡开关（中等）](https://leetcode-cn.com/problems/bulb-switcher) 假设现在总共有 16 盏灯，我们求 16 的平方根，等于 4，这就说明最后会有 4 盏灯亮着，它们分别是第 1 * 1=1 盏、第 2 * 2=4 盏、第 3 * 3=9 盏和第 4 * 4=16 盏。return (int)Math.sqrt(n);即可

## 4、其他技巧

[215. 数组中的第 K 个最大元素（中等）](https://leetcode-cn.com/problems/kth-largest-element-in-an-array) 方法一：使用最小堆，将元素挨个offer进去，每当堆的size>k，poll出一个元素，最后留下的堆顶元素即为第K个最大元素。方法二：由快排演变出的快速选择算法

[241. 为运算表达式设计优先级（中等）](https://leetcode-cn.com/problems/different-ways-to-add-parentheses) 分治法。可以以 '+' '-' '*' 这三种符号作为分隔将string分成左右两个部分，对左右两个子问题递归，并对返回的两个结果集运用 '+' '-' '*' 三种计算方式计算出新的resList，可以添加一个哈希表memo做备忘录

[1288. 删除被覆盖区间（中等）](https://leetcode-cn.com/problems/remove-covered-intervals) 将intervals数组按start升序end降序排序，左右指针记录当前大区间的start和end，若遍历到的区间end<=right，该区间需要被删除，res++；若该区间end>right，则更新左右指针为该区间，最后intervals.length - res

[56. 区间合并（中等）](https://leetcode-cn.com/problems/merge-intervals) 将intervals数组按start升序end降序排序，使用List<int[]> res记录结果，左右指针记录当前大区间的start和end，若遍历到的区间start<=right且end>right，则更新right；若该区间的start>right，说明要跳跃到新区间了，将当前left和right作为结果加入res，更新左右指针。循环结束后，不要忘记将最后一个合并区间加入res。最后返回res.toArray(new int[res.size()][])

[986. 区间列表的交集（中等）](https://leetcode-cn.com/problems/interval-list-intersections) 给的两个数组已经排好序了，使用List<int[]> res记录结果，双指针i、j分别在两个数组的区间上滑动，用a1、a2表示一区间的start和end，b1、b2表示二区间的start和end，如果b1 <= a2 && a1 <= b2，则将Math.max(a1, b1), Math.min(a2, b2)加入res。如果b2 < a2，j++，否则i++。最后返回res.toArray(new int[res.size()][])

## 5、经典面试题

[659. 分割数组为连续子序列（中等）](https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/) 使用两个HashMap，一个freq记录nums中每个元素出现的次数，一个need记录已找到的子序列还需要哪些元素，每个元素共被需要多少次。所以我们再遍历nums时，可以遇到以下几种情况：1、freq(n)==0说明n元素已经被用完，continue即可；2、need中包含n元素，且数量大于0，则freq(n)--,need(n)--,need(n+1)++；3、freq中包含n、n+1、n+2且数量都大于0，则freq(n)--,freq(n+1)--,freq(n+2)--,need(n+3)++。如果以上情况都无法遇到，返回false。此题使用Java编写时注意HashMap的取值存值方法

[牛客网 吃葡萄](https://www.nowcoder.com/questionTerminal/14c0359fb77a48319f0122ec175c9ada)

[969. 煎饼排序（中等）](https://leetcode-cn.com/problems/pancake-sorting) 方法一：找到0~n直接最大的饼i，翻转0~i，再翻转0~n，最大的饼就被放到了最下面，同样的方法再排0~n-1……即可，此方法可行，但是找到的不是最优解；

[43. 字符串相乘（中等）](https://leetcode-cn.com/problems/multiply-strings) 模拟数学的两数相乘过程，nums1的长度为n，nums2的长度为m，创建一个int[n+m]的数组res用于存储结果，倒叙遍历两个string，对两边的char转型并相乘（char可以使用 - '0'来转型），然后将结果存入res的i+j和i+j+1两个位置，注意需要累加该位置上本来存放的值。然后找到res中第一个非零的位置，从该位置开始将其转为string。需要注意处理异常场景，两个string中只要有一个"0"，结果就需要返回"0"

[224. 基本计算器（困难）](https://leetcode-cn.com/problems/basic-calculator) 使用一个全局变量index记录当前循环到的位置，构造辅助函数用递归解决括号问题，递归函数里用一个栈存储数，用一个sign记录当前符号，一个num记录遇到下一个符号前的数字，然后循环，取当前位置的字符ch后index++，使用Character.isDigit(ch)方法判断是否为数字，如果是数字，num进一位并加上ch；如果是左括号，送入递归；如果不是数字且不是空格，或已经到最后一位（因前面index++了，所以此处要判断index == s.length()），用switch..case..捕捉符号情况并将num塞入栈，然后将num重新置为0，sign=ch；如果是右括号，则break跳出循环。出循环后将栈中的数字全部相加并返回结果。

[227. 基本计算器II（中等）](https://leetcode-cn.com/problems/basic-calculator-ii) 在上一题的基础上多了*和/，只需要在switch..case..中加入 * 和 / 的情况，其他部分不变即可。注意 * 和 / 时需要将栈顶元素去出，和当前数字做相应计算后再放入栈中。

[772. 基本计算器III（困难） Plus会员](https://leetcode-cn.com/problems/basic-calculator-iii)

[42. 接雨水（困难）](https://leetcode-cn.com/problems/trapping-rain-water) 方法一：用两个备忘录分别记录i位置左侧最高和右侧最高，然后循环数组res += Math.min(l_max[i], r_max[i]) - height[i]，最后返回res。方法二：双指针left和right一头一尾，再用两个变量分别记录[0, l_max]和[r_max, n - 1]范围内的最高峰，当left < right时，l_max = Math.max(l_max, height[left])，r_max = Math.max(r_max, height[right])。如果l_max < r_max，更新res，left++；否则更新res，right--

[11. 盛最多水的容器（中等）](https://leetcode-cn.com/problems/container-with-most-water/) 双指针left和right一头一尾，循环，curArea = Math.min(height[left], height[right]) * (right - left)，res = Math.max(res, curArea)，移动左右指针高度较小的那一个

[20. 有效的括号（简单）](https://leetcode-cn.com/problems/valid-parentheses) 用一个栈记录，循环字符串中的字符，遇到左则压入栈，遇到右且栈不为空时，取栈顶元素与当前char对比是否为一对，是则将栈顶元素pop出来，不是则返回false，循环结束后判断栈是否空了，返回结果

[921. 使括号有效的最小添加（中等）](https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid) 用need和res两个变量分别存储需要的右括号数量和需要的操作次数，循环字符串中的字符，遇到左括号，则need++；遇到右括号则need--，并且判断need == -1，说明此时右括号比左括号多，需要添加一个左括号，所以res++，need重新置为0。最后返回need+res

[1541. 平衡括号串的最少插入（中等）](https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string) 每个左括号要匹配两个右括号。用need和res两个变量分别存储需要的右括号数量和需要的操作次数，循环字符串中的字符，遇到左括号，则need+2，并且判断need%2是否余数为1，若是则说明右括号的数量不够，需要插入一个右括号res++，need--；如果遇到右括号则need--，并且判断need == -1，说明需要插入一个左括号res++，同时将need置为1。最后返回need+res

[391. 完美矩形（困难）](https://leetcode-cn.com/problems/perfect-rectangle/) 核心思想是判断面积和四角。1、每个小矩阵的面积和要等于最后大矩阵的面积；2、大矩阵内部的每个内部角出现次数应该为偶数，所以用一个Set记录，没有出现则add，出现过了则remove，最后一定留下四个，且这四个应为最后大矩阵的四个外部角。思路不难，但是注意对set的处理，一开始使用了HashSet<int[]>的结构，但是不知道为什么总报错，后来使用了HashSet<String>的结构，将顶点的坐标处理成字符串x + "," + y存入则没有问题，不知道是不是底层什么缘故

[855. 考场就座（中等）](https://leetcode-cn.com/problems/exam-room) 思路不复杂，但是要写的内容和要注意的细节特别多。题目给了一个初始化函数ExamRoom，一个分配座位的函数seat，一个离开座位的函数leave。解决思路是，将连续的空座位看做线段，比如p序列的座位上坐了人，用一个HashMap<Integer, int[]>startMap存储以p为左端点的线段，用另一个HashMap<Integer, int[]>endMap存储以p为右端点的线段，用一个TreeSet<int[]>pq存储所有线段，TreeSet是一个有序集合，可以快速查找最值，也可以remove指定元素，满足我们的需求。所以在初始化函数中我们要初始这几个数据结构，注意pq的比较器应该是disA - disB，如果相等b[0] - a[0]将小的序列放在前面，并且插入{-1, N}数组表示一开始全场为空的情况。我们还需要几个辅助函数：addLine函数向pq中add线段line，向startMap中put(line[0], line)，向endMap中put(line[1], line)；removeLine函数从pq中remove线段line，从startMap中remove line[0]，从endMap中remove line[1]；distance函数用于计算线段的长度，因为题目要求多个选择时选择索引最小的那个，所以该函数不能单纯的只计算距离，而是应该计算(y - x) / 2，x == -1时返回y，y == N时返回N - 1 - x。然后就可以写seat和leave函数了。seat函数从pq中取出最长的数组，选取(y - x) / 2 + x作为座位，以此将最长数组一分为二，remove最长的，add左右两段。leave函数则是根据座位p取出左右两段，合二为一，add新数组，remove左右两段

[392. 判断子序列（简单）](https://leetcode-cn.com/problems/is-subsequence) 方法一：双指针，i和j分别在字符串s和t上滑动，j++，s[i]==t[j]则i++，最后判断i==s.length()。方法二：适用于该问题的follow up。用一个ArrayList<Integer>[] 类型的数组index存储t中各个字符所在索引，用一个变量j在t上滑动，遍历s，若当前遍历到的字符c不在index中，即index[c] == null，则返回false，否则调用辅助函数二分查找大于j的最小值，该最小值若超过了index[c]的范围则说明没有满足匹配条件的字符，返回false，否则将j更新为index[c].get(left) + 1