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

[297. 二叉树的序列化和反序列化（困难）](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree) 前中后序遍历三种解法

[341. 扁平化嵌套列表迭代器（中等）](https://leetcode-cn.com/problems/flatten-nested-list-iterator)

[236. 二叉树的最近公共祖先（中等）](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

[222. 完全二叉树的节点个数（中等）](https://leetcode-cn.com/problems/count-complete-tree-nodes)

## 5、二叉搜索树（BST）

[230. BST第K小的元素（中等）](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst) 两个变量记录节点val和节点序号，中序遍历到第K个

[538. 二叉搜索树转化累加树（中等）](https://leetcode-cn.com/problems/convert-bst-to-greater-tree) 中序遍历变形，右中左，使树降序输出，记录sum并赋值

[1038. BST转累加树（中等）](https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree) 同上

[700. 二叉搜索树中的搜索（简单）](https://leetcode-cn.com/problems/search-in-a-binary-search-tree) 递归，=、<、> （BST增删改查的框架）

[701. 二叉搜索树中的插入操作（中等）](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree) 遇空插入

[450. 删除二叉搜索树中的节点（中等）](https://leetcode-cn.com/problems/delete-node-in-a-bst) 找到后，若单侧孩子为空，直接将另一侧提上来，若两侧均有孩子，找到右孩子的min提上来，右孩子继续调用删除方法

[98. 验证二叉搜索树（中等）](https://leetcode-cn.com/problems/validate-binary-search-tree) 递归，root min max

[96. 不同的二叉搜索树（简单）](https://leetcode-cn.com/problems/unique-binary-search-trees)

[95. 不同的二叉搜索树II（中等）](https://leetcode-cn.com/problems/unique-binary-search-trees-ii)

[1373. 二叉搜索子树的最大键值和（困难）](https://leetcode-cn.com/problems/maximum-sum-bst-in-binary-tree)

## 6、图论

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

## 7、设计数据结构

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

## 8、数组