# 做过的题分类

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

[114. 二叉树展开为链表（中等）](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list) 递归，先将左右孩子分别拉平，然后左变右，右接到新右的叶子节点上

[116. 填充每个节点的下一个右侧节点指针（中等）](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node) 辅助函数递归，左指向右，1的右指向2的左

[654. 最大二叉树（中等）](https://leetcode-cn.com/problems/maximum-binary-tree/) 辅助函数递归，在每段内找到最大值构造为root

[105. 从前序与中序遍历序列构造二叉树（中等）](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)辅助函数递归，找到root和左右孩子区间

[106. 从中序与后序遍历序列构造二叉树（中等）](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)同上

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

[1109. 航班预订统计（中等）](https://leetcode-cn.com/problems/corporate-flight-bookings/) 构建差额数组

[1094. 拼车（中等）](https://leetcode-cn.com/problems/car-pooling/) 同上

[167. 两数之和 II - 输入有序数组（简单）](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted) 左右指针

[344. 反转字符串（简单）](https://leetcode-cn.com/problems/reverse-string/) 左右指针

[76. 最小覆盖子串（困难）](https://leetcode-cn.com/problems/minimum-window-substring) 滑动窗口，两个map分别记录need和window，一个int记录valid，每满足一个字符valid++。注意字符串的处理，注意细节即可

[567. 字符串的排列（中等）](https://leetcode-cn.com/problems/permutation-in-string) 滑动窗口，注意窗口大小可以固定，约束条件可以优化

[438. 找到字符串中所有字母异位词（中等）](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string) 滑动窗口，同上

[3. 无重复字符的最长子串（中等）](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters) 滑动窗口，只需要一个map记录window即可，内层循环条件改为window.get(i) > 1，外层循环更新res

[704. 二分查找（简单）](https://leetcode-cn.com/problems/binary-search)

[34. 在排序数组中查找元素的第一个和最后一个位置（中等）](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

[875. 爱吃香蕉的珂珂（中等）](https://leetcode-cn.com/problems/koko-eating-bananas/)

[1011. 在D天内送达包裹的能力（中等）](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/)

[410. 分割数组的最大值（困难）](https://leetcode-cn.com/problems/split-array-largest-sum/)

[870. 优势洗牌（中等）](https://leetcode-cn.com/problems/advantage-shuffle)

[380. 常数时间插入、删除和获取随机元素（中等）](https://leetcode-cn.com/problems/insert-delete-getrandom-o1)

[710. 黑名单中的随机数（困难）](https://leetcode-cn.com/problems/random-pick-with-blacklist)

[316. 去除重复字母（中等）](https://leetcode-cn.com/problems/remove-duplicate-letters)

[1081. 不同字符的最小子序列（中等）](https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters)

[26. 删除有序数组中的重复项（简单）](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

[83. 删除排序链表中的重复元素（简单）](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

[27. 移除元素（简单）](https://leetcode-cn.com/problems/remove-element/)

[283. 移动零（简单）](https://leetcode-cn.com/problems/move-zeroes/)

[1. 两数之和（简单）](https://leetcode-cn.com/problems/two-sum)

[170. 两数之和 III - 数据结构设计（简单）](https://leetcode-cn.com/problems/two-sum-iii-data-structure-design)

## 7、图论

[797. 所有可能的路径（中等）](https://leetcode-cn.com/problems/all-paths-from-source-to-target/)

[207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

[210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

[785. 判断二分图（中等）](https://leetcode-cn.com/problems/is-graph-bipartite)

[886. 可能的二分法（中等）](https://leetcode-cn.com/problems/possible-bipartition)

[323. 无向图中的连通分量数目（中等）](https://leetcode-cn.com/problems/number-of-connected-components-in-an-undirected-graph/)

[130. 被围绕的区域（中等）](https://leetcode-cn.com/problems/surrounded-regions/)

[990. 等式方程的可满足性（中等）](https://leetcode-cn.com/problems/satisfiability-of-equality-equations/)

[261. 以图判树（中等）](https://leetcode-cn.com/problems/graph-valid-tree/)

[1135. 最低成本联通所有城市（中等）](https://leetcode-cn.com/problems/connecting-cities-with-minimum-cost/)

[1584. 连接所有点的最小费用（中等）](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

[277. 搜索名人（中等）](https://leetcode-cn.com/problems/find-the-celebrity/)

[743. 网络延迟时间（中等）](https://leetcode-cn.com/problems/network-delay-time)

[1514. 概率最大的路径（中等）](https://leetcode-cn.com/problems/path-with-maximum-probability)

[1631. 最小体力消耗路径（中等）](https://leetcode-cn.com/problems/path-with-minimum-effort)

## 8、设计数据结构

[146. LRU缓存机制（中等）](https://leetcode-cn.com/problems/lru-cache/)

[460. LFU缓存机制（困难）](https://leetcode-cn.com/problems/lfu-cache/)

[895. 最大频率栈（困难）](https://leetcode-cn.com/problems/maximum-frequency-stack/)

[295. 数据流的中位数（困难）](https://leetcode-cn.com/problems/find-median-from-data-stream)

[355. 设计推特（中等）](https://leetcode-cn.com/problems/design-twitter)

[496. 下一个更大元素I（简单）](https://leetcode-cn.com/problems/next-greater-element-i)

[503. 下一个更大元素II（中等）](https://leetcode-cn.com/problems/next-greater-element-ii)

[739. 每日温度（中等）](https://leetcode-cn.com/problems/daily-temperatures/)

[239. 滑动窗口最大值（困难）](https://leetcode-cn.com/problems/sliding-window-maximum)

[232. 用栈实现队列（简单）](https://leetcode-cn.com/problems/implement-queue-using-stacks)

[225. 用队列实现栈（简单）](https://leetcode-cn.com/problems/implement-stack-using-queues)
