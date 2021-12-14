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

[5. 最长回文子串（中等）](https://leetcode-cn.com/problems/longest-palindromic-substring) for循环从头开始，从中心向两边扩散，不断更新答案，注意奇偶两种情况，注意防止越界。其他思路：[动规]() 、[马拉车算法]()

[234. 回文链表（简单）](https://leetcode-cn.com/problems/palindrome-linked-list)