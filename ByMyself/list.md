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

[652. 寻找重复的子树（中等）](https://leetcode-cn.com/problems/find-duplicate-subtrees)

[297. 二叉树的序列化和反序列化（困难）](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree) 前中后序遍历三种解法

[341. 扁平化嵌套列表迭代器（中等）](https://leetcode-cn.com/problems/flatten-nested-list-iterator)

[236. 二叉树的最近公共祖先（中等）](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

[222. 完全二叉树的节点个数（中等）](https://leetcode-cn.com/problems/count-complete-tree-nodes)

## 5、二叉搜索树（BST）

[230. BST第K小的元素（中等）](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst)

[538. 二叉搜索树转化累加树（中等）](https://leetcode-cn.com/problems/convert-bst-to-greater-tree)

[1038. BST转累加树（中等）](https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree)

[450. 删除二叉搜索树中的节点（中等）](https://leetcode-cn.com/problems/delete-node-in-a-bst)

[701. 二叉搜索树中的插入操作（中等）](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree)

[700. 二叉搜索树中的搜索（简单）](https://leetcode-cn.com/problems/search-in-a-binary-search-tree)

[98. 验证二叉搜索树（中等）](https://leetcode-cn.com/problems/validate-binary-search-tree)

[96. 不同的二叉搜索树（简单）](https://leetcode-cn.com/problems/unique-binary-search-trees)

[95. 不同的二叉搜索树II（中等）](https://leetcode-cn.com/problems/unique-binary-search-trees-ii)

[1373. 二叉搜索子树的最大键值和（困难）](https://leetcode-cn.com/problems/maximum-sum-bst-in-binary-tree)

## 6、图论