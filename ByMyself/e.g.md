# 一些Java语法示例

## for循环

```java
for(int i = 0; i < nums.length; i++)
```

## 增强型for循环

```java
for (声明语句 : 表达式)
// 声明语句：声明新的局部变量，该变量的类型必须和数组元素的类型匹配。其作用域限定在循环语句块，其值与此时数组元素的值相等。
// 表达式：表达式是要访问的数组名，或者是返回值为数组的方法。

// 例如
for (ListNode head : lists) {
    if (head != null)
        pq.add(head);
}
```

## Lambda表达式

```java
(方法参数) -> {方法要实现的内容}
```

## 三元表达式

```java
int i = l1 != null ? l1.val : 0;
```

## int
```java
int i = Integer.parseInt("123");
```

## 数组

```java
// 创建
int[] myArray = new int[10];
return new int[]{i, j};
```

## 字符串

```java
// 创建
String str = new String("hello");
// 截取，注意左闭右开
String s2 = str.substring(0, 4)
str.split("l");
str.equals("world");
```

## StringBuilder

```java
StringBuilder sb = new StringBuilder();
sb.append("t");
sb.toString();
```

## 链表

```java
List<Integer> list = new LinkedList<>();
list.add(); list.remove(0); list.addFirst(123);
list.isEmpty(); list.size();
```

## 哈希表

```java
// 创建
HashMap<Integer, Integer> map = new HashMap<>();
HashMap<Character, Integer> window = new HashMap<>();
// 查找-返回布尔
map.containsKey(tmp);
map.containsValue(tmp);
map.get(tmp);
map.put(nums[i], i);
map.remove(1);
// 查找第一个参数对应的val，有则返回val，没有的话则返回第二个参数
window.getOrDefault(a, 0);
// 如果map中没有第一个参数为key的键值对，则插入以第一和第二个参数为K-V的键值对
map.putIfAbsent(1, 1)
```

## LinkedHashMap

LinkedHashMap是继承于HashMap，是基于HashMap和双向链表来实现的，有序

```java
LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();
cache.put(key, val);
cache.get(key);
cache.containsKey(key);
cache.remove(key);
cache.size();
cache.keySet(); // 返回key集合
```

## LinkedHashSet

```java
LinkedHashSet<Integer> keys = new LinkedHashSet<>();
keys.iterator().next();
```

## 集合

Set接口继承于Collection接口，是一个不允许出现重复元素，并且无序的集合，主要有HashSet和TreeSet两大实现类。

```java
Set<Integer> set = new HashSet<>();
set.iterator(); // 遍历set
```

## 栈

```java
Stack<Integer> stack = new Stack<>();
stack.push();
stack.pop();
stack.peek();
```

## 队列

```java
Queue<Integer> queue = new LinkedList<>();
queue.add(); queue.offer();
queue.poll(); queue.peek();
queue.size();
```

## 优先队列（最小堆）

```java
// 创建
PriorityQueue<Integer> numbers = new PriorityQueue<>();
// 这种创建，使用lambda表达式复写了优先队列的compare方法
PriorityQueue<ListNode> pq = new PriorityQueue<>(lists.length, (a, b)->(a.val - b.val));
// 等同于
PriorityQueue<ListNode> pq = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
    @Override
    public int compare (ListNode a, ListNode b) {
        // 优先队列为最小堆，需要小的元素在前，所以a.val比b.val小时，返回负数（false），表示不用交换
        if (a.val < b.val) {
            return -1;
        } else if (a.val == b.val) {
            return 0;
        }
        return 1;
    }
});

// 关于Comparator
// 最小堆
int compare (Integer a, Integer b) {
    return a < b ? -1 : 1; // return a.compareTo(b);
}
// 最大堆
int compare (Integer a, Integer b) {
    return a > b ? -1 : 1;
}

// 插入
PriorityQueue.add();
// 弹出（最小值）
PriorityQueue.poll();
```