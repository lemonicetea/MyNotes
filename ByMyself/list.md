# 做过的数据结构题分类（87题）

## 1、单链表常规思路

[21. 合并两个有序链表（简单）](https://leetcode-cn.com/problems/merge-two-sorted-lists/) 双指针

[23. 合并K个升序链表（困难）](https://leetcode-cn.com/problems/merge-k-sorted-lists/) 优先队列

[141. 环形链表（简单）](https://leetcode-cn.com/problems/linked-list-cycle/) 快慢指针，相遇

[142. 环形链表 II（中等）](https://leetcode-cn.com/problems/linked-list-cycle-ii/) 快慢指针，相遇一个回起点，同速再次相遇即为环起点

[876. 链表的中间结点（简单）](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 快慢指针，两倍速

[160. 相交链表（简单）](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/) 双指针，两条链表相连直到指针相等（null也是相等）

[19. 删除链表的倒数第 N 个结点（中等）](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) 双指针

## 2、翻转链表

[206. 反转链表（简单）](https://leetcode-cn.com/problems/reverse-linked-list/) 递归，单个或三个指针

[92. 反转链表II（中等）](https://leetcode-cn.com/problems/reverse-linked-list-ii/) 递归

[25. K个一组翻转链表（困难）](https://leetcode-cn.com/problems/reverse-nodes-in-k-group) 双递归，三个指针返回新头，base case链表长度小于k

## 3、回文

[5. 最长回文子串（中等）](https://leetcode-cn.com/problems/longest-palindromic-substring) 1、for循环从头开始，从中心向两边扩散，不断更新答案，注意奇偶两种情况，注意防止越界；2、动规 ；3、马拉车算法。

[234. 回文链表（简单）](https://leetcode-cn.com/problems/palindrome-linked-list) 1、新造一条翻转链表比较；2、递归，链表后序遍历比较；3、找到中间结点，翻转后半链表比较。

## 4、二叉树

[226. 翻转二叉树（简单）](https://leetcode-cn.com/problems/invert-binary-tree) 递归，交换左右孩子

[114. 二叉树展开为链表（中等）](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list) 递归，先将左右孩子分别拉平，然后左变右，不停向下找到叶子节点，将右半截子接到叶子节点上

[116. 填充每个节点的下一个右侧节点指针（中等）](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node) 辅助函数递归，传入相邻的两个节点，1指向2，然后递归调用：1的左和右，2的左和右，1的右和2的左

[654. 最大二叉树（中等）](https://leetcode-cn.com/problems/maximum-binary-tree/) 辅助函数递归，在每段内for循环找到最大值构造为root，对左右孩子分别继续调用递归函数

[105. 从前序与中序遍历序列构造二叉树（中等）](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)辅助函数递归，传入int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd六个参数，preorder的preStart位置是root，在inorder中找到root的索引，切割出左右孩子继续递归

[106. 从中序与后序遍历序列构造二叉树（中等）](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)同上，postorder的postEnd位置是root

[652. 寻找重复的子树（中等）](https://leetcode-cn.com/problems/find-duplicate-subtrees) HashMap记录每个节点为根后序遍历拼接出的字符串和出现次数

[297. 二叉树的序列化和反序列化（困难）](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree) 三种解法，前序、后续变形（中右左）、层序

[341. 扁平化嵌套列表迭代器（中等）](https://leetcode-cn.com/problems/flatten-nested-list-iterator) 主要实现hasNext方法，不难，但是比较有代表性

[236. 二叉树的最近公共祖先（中等）](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) 递归，处理三种情况

[222. 完全二叉树的节点个数（中等）](https://leetcode-cn.com/problems/count-complete-tree-nodes) 递归简单count，通过增加判断左右子树同等高度时节点总数为2^h-1来缩短时间复杂度

## 5、二叉搜索树（BST）

[230. BST第K小的元素（中等）](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst) 两个变量记录节点val和节点序号，中序遍历到第K个

[538. 二叉搜索树转化累加树（中等）](https://leetcode-cn.com/problems/convert-bst-to-greater-tree) 中序遍历变形，右中左，使树降序输出，记录sum并赋值

[1038. BST转累加树（中等）](https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree) 同上

[700. 二叉搜索树中的搜索（简单）](https://leetcode-cn.com/problems/search-in-a-binary-search-tree) 递归，=、<、> （BST增删改查的框架）

[701. 二叉搜索树中的插入操作（中等）](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree) 遇空插入

[450. 删除二叉搜索树中的节点（中等）](https://leetcode-cn.com/problems/delete-node-in-a-bst) 找到后，若单侧孩子为空，直接将另一侧提上来，若两侧均有孩子，找到右孩子的min提上来，右孩子继续调用删除方法

[98. 验证二叉搜索树（中等）](https://leetcode-cn.com/problems/validate-binary-search-tree) 递归，root min max

[96. 不同的二叉搜索树（中等）](https://leetcode-cn.com/problems/unique-binary-search-trees) 递归，该节点可能总数=左子树可能总数*右子树可能总数

[95. 不同的二叉搜索树II（中等）](https://leetcode-cn.com/problems/unique-binary-search-trees-ii) 辅助函数递归，每次新建一个list（确保主函数取到的是最外层完整的树），三层for循环

[1373. 二叉搜索子树的最大键值和（困难）](https://leetcode-cn.com/problems/maximum-sum-bst-in-binary-tree) 辅助函数后序遍历递归，记录当前节点int[isBST, minVlue, maxVlue, sum]，是BST的话更新maxSum值，注意maxSum初始化为0（默认null节点最大和为0）

## 6、数组

[303. 区域和检索 - 数组不可变（中等）](https://leetcode-cn.com/problems/range-sum-query-immutable) 构建长度大一的preSum记录前缀和

[304. 二维区域和检索 - 矩阵不可变（中等）](https://leetcode-cn.com/problems/range-sum-query-2d-immutable) 构建[x+1][y+1]的preSum，记录从(1,1)到(m,n)的前缀和，注意推导公式

[560. 和为K的子数组（中等）](https://leetcode-cn.com/problems/subarray-sum-equals-k) 暴力循环。follow up：构建HashMap

[370. 区间加法（中等）PLUS会员](https://leetcode-cn.com/problems/range-addition/)

[1109. 航班预订统计（中等）](https://leetcode-cn.com/problems/corporate-flight-bookings/) 构建差分数组

[1094. 拼车（中等）](https://leetcode-cn.com/problems/car-pooling/) 构建差分数组

[167. 两数之和 II - 输入有序数组（简单）](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted) 左右指针

[344. 反转字符串（简单）](https://leetcode-cn.com/problems/reverse-string/) 左右指针

[76. 最小覆盖子串（困难）](https://leetcode-cn.com/problems/minimum-window-substring) 滑动窗口，两个map分别记录need和window，一个int记录valid，每满足一个字符valid++。注意字符串的处理，注意细节即可

[567. 字符串的排列（中等）](https://leetcode-cn.com/problems/permutation-in-string) 滑动窗口，注意窗口大小可以固定，约束条件可以优化

[438. 找到字符串中所有字母异位词（中等）](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string) 滑动窗口，同上

[3. 无重复字符的最长子串（中等）](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters) 滑动窗口，只需要一个map记录window即可，内层循环条件改为window.get(i) > 1，外层循环更新res

[704. 二分查找（简单）](https://leetcode-cn.com/problems/binary-search) 二分查找

[34. 在排序数组中查找元素的第一个和最后一个位置（中等）](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) 二分查找，两个辅助函数左右边界

[875. 爱吃香蕉的珂珂（中等）](https://leetcode-cn.com/problems/koko-eating-bananas/) 二分查找，注意找到f(x)和target

[1011. 在D天内送达包裹的能力（中等）](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/) 二分查找，注意找到f(x)和target

[410. 分割数组的最大值（困难）](https://leetcode-cn.com/problems/split-array-largest-sum/) 二分查找，子数组和的最大值和分割数量是单调递减关系

[870. 优势洗牌（中等）](https://leetcode-cn.com/problems/advantage-shuffle) n1升序排序，n2存入最大优先队列，n1双指针，用最大比最大，比不过用最小换人头

[380. 常数时间插入、删除和获取随机元素（中等）](https://leetcode-cn.com/problems/insert-delete-getrandom-o1) 用list实现，HashMap作为辅助记录val-index使得查找降为o(1)，删除时注意各语句执行顺序

[710. 黑名单中的随机数（困难）](https://leetcode-cn.com/problems/random-pick-with-blacklist) [0, end)为白名单，[end, n)为黑名单，若黑名单出现在了白名单的区域内，用map记录，将val指向黑名单区域内，通过映射关系将白名单紧凑在前区间内

[316. 去除重复字母（中等）](https://leetcode-cn.com/problems/remove-duplicate-letters) 用stack存储非重复字符，利用字符的ASCII码，构造int[] count和boolean[] instack，入栈时与栈顶元素比较大小，最后构造StringBuilder并反转即可。最终得到遵循ASCII码大小且相对顺序不变的子序列

[1081. 不同字符的最小子序列（中等）](https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters) 同316题，一套代码

[26. 删除有序数组中的重复项（简单）](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/) 快慢指针

[83. 删除排序链表中的重复元素（简单）](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/) 快慢指针

[27. 移除元素（简单）](https://leetcode-cn.com/problems/remove-element/) 快慢指针

[283. 移动零（简单）](https://leetcode-cn.com/problems/move-zeroes/) 快慢指针，前面部位，最后剩下的全部填入零

[1. 两数之和（简单）](https://leetcode-cn.com/problems/two-sum) 构造HashMap记录val-index

[170. 两数之和 III - 数据结构设计（简单）PLUS会员](https://leetcode-cn.com/problems/two-sum-iii-data-structure-design) 

## 7、设计数据结构

[146. LRU缓存机制（中等）](https://leetcode-cn.com/problems/lru-cache/) Least Recently Used。淘汰时间最远的数据。借助一个哈希双向链表LinkedHashMap和一个int常量

[460. LFU缓存机制（困难）](https://leetcode-cn.com/problems/lfu-cache/) Least Frequently Used。淘汰访问频次最低的数据，频次最低的有多条，则淘汰最旧的。使用到了不常见的数据结构：链表哈希集合LinkedHashSet。借助3个HashMap（KV表、KF表、FKs表，Ks使用LHS集合）和2个int常量（记录容量和最小频率）

[895. 最大频率栈（困难）](https://leetcode-cn.com/problems/maximum-frequency-stack/) 借助2个HashMap（VF表、FVs表，Vs使用stack）和1个int常量

[295. 数据流的中位数（困难）](https://leetcode-cn.com/problems/find-median-from-data-stream) 一个最大堆和一个最小堆，控制两边size差值在1以内

[355. 设计推特（中等）](https://leetcode-cn.com/problems/design-twitter) 需要设计Tweet、User两个私有类，按时间线展示动态的方法里需要运用到合并K个有序链表的技巧，其他没有难度，注意细节

[496. 下一个更大元素I（简单）](https://leetcode-cn.com/problems/next-greater-element-i) 单调栈，倒序循环数组，使用栈暂存和比较内容

[503. 下一个更大元素II（中等）](https://leetcode-cn.com/problems/next-greater-element-ii) 单调栈，2倍长度倒序循环数组，模除取余数存入对应的index中

[739. 每日温度（中等）](https://leetcode-cn.com/problems/daily-temperatures/) 单调栈，NGE模型的实际应用，换汤不换药

[239. 滑动窗口最大值（困难）](https://leetcode-cn.com/problems/sliding-window-maximum) 实现一个MonotonicQueue单调队列数据结构（基于LinkedList双向链表），用于作为滑动窗口

[232. 用栈实现队列（简单）](https://leetcode-cn.com/problems/implement-queue-using-stacks) 用两个stack实现，s1队尾，s2队头

[225. 用队列实现栈（简单）](https://leetcode-cn.com/problems/implement-stack-using-queues) 一个队列，一个变量记录最后入队的元素值，每次pop按顺序将元素从队头取出重新塞入队尾，变量记录倒数第二个元素

## 8、图论

[797. 所有可能的路径（中等）](https://leetcode-cn.com/problems/all-paths-from-source-to-target/) 从0到n-1的所有可能路径，使用一个List<List<Integer>> res记录所有结果，写一个辅助函数traverse用于递归，添加节点p到路径中，如果p==n-1，则将path加入res中，并移除path末尾元素，返回。否则遍历p的下个节点，递归调用。循环结束移除path末尾元素

[207. 课程表](https://leetcode-cn.com/problems/course-schedule/) 首先我们需要一个辅助函数buildGraph将记录前置课程的int[][] prerequisites转化为有向图List<Integer>[] graph，将graph长度初始化为numCourses，对每个位置初始化一个list，然后遍历prerequisites，将p[0]记录为to，p[1]记录为from，graph[from].add(to)。然后我们需要一个辅助函数traverse用来遍历图，创建全局变量boolean hasCycle记录是否存在环，boolean[] visited记录走过的节点，boolean[] onPath记录当前正在遍历的路径，函数中首先判断onPath[s]，true说明有环，然后判断visited[s] || hasCycle，true可以直接返回，然后将visited[s]和onPath[s]分别置为true，遍历graph[s]并递归调用traverse，出循环后将onPath[s]置为false。那么在主函数canFinish中，初始化各变量，遍历0~numCourses-1，调用traverse，最后判断hasCycle

[210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/) 在上一题的基础上，增加一个List<Integer> postorder用于记录遍历顺序，因为是上课顺序，所以需要改造部分逻辑，将buildGraph函数中图的打印换成graph[to].add(from)。在traverse函数中，使用后序遍历逻辑，在递归完子节点后postorder.add(s)。在主函数中判断环，有环则返回空数组，无环则将postorder从链表转换为数组再返回

[785. 判断二分图（中等）](https://leetcode-cn.com/problems/is-graph-bipartite) 可以用BFS和DFS两种方法做。用一个全局变量res存储结果，一个boolean数组存储visited，一个boolean数组存储colors。在主函数中初始化各全局变量，然后遍历0~n-1，对每个未被visited标记的节点调用traverse/BFS函数，以防存在散落的节点。在traverse函数中，如果!res直接返回，否则标记visited，遍历graph[s]，对每个与s连接的节点v进行判断，如果v没有被visited，将v染色为!colors[s]，并traverse(v)，如果被visited，判断colors[v] == colors[s]，相同则说明无法形成二分图，将res置为false

[886. 可能的二分法（中等）](https://leetcode-cn.com/problems/possible-bipartition) 在上一题的基础上增加一个构建无向图的辅助函数buildGraph即可，其他一样。注意graph的两种构建方式，一种是邻接表，一种是邻接矩阵，我们这里使用邻接表，用数据结构List<Integer>[]存储

[323. 无向图中的连通分量数目（中等）PLUS会员](https://leetcode-cn.com/problems/number-of-connected-components-in-an-undirected-graph/) 动态连通性问题，使用Union-Find算法。

[130. 被围绕的区域（中等）](https://leetcode-cn.com/problems/surrounded-regions/)

[990. 等式方程的可满足性（中等）](https://leetcode-cn.com/problems/satisfiability-of-equality-equations/)

[261. 以图判树（中等）](https://leetcode-cn.com/problems/graph-valid-tree/)

[1135. 最低成本联通所有城市（中等）](https://leetcode-cn.com/problems/connecting-cities-with-minimum-cost/)

[1584. 连接所有点的最小费用（中等）](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

[277. 搜索名人（中等）](https://leetcode-cn.com/problems/find-the-celebrity/)

[743. 网络延迟时间（中等）](https://leetcode-cn.com/problems/network-delay-time)

[1514. 概率最大的路径（中等）](https://leetcode-cn.com/problems/path-with-maximum-probability)

[1631. 最小体力消耗路径（中等）](https://leetcode-cn.com/problems/path-with-minimum-effort)