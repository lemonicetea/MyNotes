# 一、二分查找

## 1.搜索

```java
// 搜索旋转排序数组
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) return -1;
        int left = 0, right = nums.length - 1;
        while(left < right) {
            int mid = left + ((right - left) >> 1);
            if(nums[mid] == target) return mid;
            // 比较mid与right的值判断有序段
            // 除非left == right，否则mid与right不会是同一个位置
            if(nums[mid] < nums[right]) {
                // 右半边有序
                if(target > nums[mid] && target <= nums[right])
                    left = mid + 1;
                else
                    right = mid - 1;
            } else {
                // 左半边有序
                if(target >= nums[left] && target < nums[mid])
                    right = mid - 1;
                else
                    left = mid + 1;
            }
        }
        return nums[left] == target ? left : -1;
    }
}

// 搜索旋转排序数组（含重复元素）
class Solution {
    public boolean search(int[] nums, int target) {
        if(nums.length == 0) return false;
        int left = 0, right = nums.length - 1;
        while(left < right) {
            int mid = left + ((right - left) >> 1);
            if(nums[mid] == target) return true;
            if(nums[mid] == nums[right]) --right;
            else if(nums[mid] < nums[right]) {
                if(target > nums[mid] && target <= nums[right])
                    left = mid + 1;
                else
                    right = mid - 1;
            } else {
                if(target >= nums[left] && target < nums[mid])
                    right = mid - 1;
                else
                    left = mid + 1;
            }
        }
        return nums[left] == target ? true : false;
    }
}

// 寻找旋转排序数组的最小值（无重复元素）
class Solution {
    public int findMin(int[] nums) {
        if(nums.length == 0) return -1;
        int l = 0, r = nums.length - 1;
        while(l < r) {
            if(nums[l] < nums[r])
                return nums[l];
            int m = l + ((r - l) >> 1);
            if(nums[m] < nums[r]) // 右半边有序
                r = m; // 因为m小，所以m可能是最小值
            else
                l = m + 1;
        }
        return nums[l];
    }
}

// 寻找旋转排序数组的最小值（含重复元素）
class Solution {
    public int findMin(int[] nums) {
        if(nums.length == 0) return -1;
        int l = 0, r = nums.length - 1;
        while(l < r) {
            if(nums[l] < nums[r]) return nums[l];
            if(nums[l] == nums[r]) {
                --r;
                continue;
            }
            // nums[l] > nums[r] 即当前数组为旋转数组
            int m = l + ((r - l) >> 1);
            if(nums[m] <= nums[r]) // 右半边升序，m可能是最小值
                r = m;
            else // 左半边升序，m肯定不是最小值
                l = m + 1;
        }
        return nums[l];
    }
}

// 在排序数组中查找元素的第一个和最后一个位置
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1, -1};
        if(nums.length == 0) return res;
        // 二分法寻找第一次出现的位置
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int m = l + ((r - l) >> 1);
            if(nums[m] < target)
                l = m + 1;
            else
                r = m;
        }
        if(nums[l] != target) return res;
        res[0] = l;
        // 二分法寻找最后一次出现的位置的后一个位置
        r = nums.length;
        while(l < r) {
            int m = l + ((r - l) >> 1);
            if(nums[m] < target + 1)
                l = m + 1;
            else
                r = m;
        }
        res[1] = l - 1;
        return res;
    }
}

// 搜索插入位置
class Solution {
    public int searchInsert(int[] nums, int target) {
        if(nums == null || nums.length == 0)
            return 0;
        int l = 0, r = nums.length;
        while(l < r) {
            int m = l + ((r - l) >> 1);
            if(nums[m] == target)
                return m;
            if(nums[m] < target) l = m + 1;
            else r = m;
        }
        return l;
    }
}

// 寻找峰值
class Solution {
    public int findPeakElement(int[] nums) {
        if(nums.length == 1) return 0;
        int left = 0, right = nums.length - 1;
        while(left < right) {
            int mid = (left + right) >> 1;
            if(nums[mid] < nums[mid + 1])
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }
}

// 寻找重复数
class Solution {
    public int findDuplicate(int[] nums) {
        int left = 1, right = nums.length;
        while(left < right) {
            int mid = left + ((right - left) >> 1);

            int count = 0;
            for(int num : nums)
                if(num <= mid) ++count;
            
            if(count > mid)
                right = mid;
            else
                left = mid + 1;
        }
        return left;
    }
}

// 搜索二维矩阵
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length == 0)
            return false;
        
        int m = matrix.length, n = matrix[0].length;
        if(target < matrix[0][0] || target > matrix[m - 1][n - 1])
            return false;
        
        // 确定行
        int left = 0, right = m - 1;
        while(left <= right) {
            if(left == right) {
                if(target < matrix[left][0] || target > matrix[left][n - 1])
                    return false;
                break;
            }
            
            int mid = left + ((right - left) >> 1);
            if(target == matrix[mid][0] || target == matrix[mid][n - 1])
                return true;
            
            if(target < matrix[mid][0])
                right = mid - 1;
            else if(target > matrix[mid][n - 1])
                left = mid + 1;
            else
                break;
        }
        int targetLine = left + ((right - left) >> 1);
        // 确定列
        left = 0;
        right = n - 1;
        while(left <= right) {
            int mid = left + ((right - left) >> 1);
            if(target == matrix[targetLine][mid])
                return true;
            if(target > matrix[targetLine][mid])
                left = mid + 1;
            else
                right = mid - 1;
        }
        
        return false;
    }
}

// 搜索二维矩阵II
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length, n = matrix[0].length;
        int i = 0, j = n - 1;
        while(i < m && j >= 0) {
            if(matrix[i][j] == target)
                return true;
            else if(matrix[i][j] > target)
                --j;
            else
                ++i;
        }
        return false;
    }
}

// 有序矩阵中第K小的元素
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return -1;
        int n = matrix.length;
        int left = matrix[0][0], right = matrix[n - 1][n - 1];
        while(left < right) {
            int mid = left + (right - left) / 2;
            int count = countLessThan(matrix, mid);
            if(count < k)
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }

    private int countLessThan(int[][] matrix, int target) {
        int n = matrix.length;
        int row = 0, col = n - 1;
        int count = 0;
        while(row < n && col >= 0) {
            if(matrix[row][col] <= target) {
                count += col + 1;
                ++row;
            } else
                --col;
        }
        return count;
    }
}

// 找到K个最接近x的元素
class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> res = new ArrayList<>();
        int minIdx = 0, minAvg = Integer.MAX_VALUE;
        for(int i = 0; i < arr.length; ++i) {
            if(Math.abs(arr[i] - x) < minAvg) {
                minIdx = i;
                minAvg = Math.abs(arr[i] - x);
            }
        }
        int i = minIdx - 1, j = minIdx + 1;
        for( ; k > 1; --k) {
            int left = i >= 0 ? Math.abs(arr[i] - x) : Integer.MAX_VALUE;
            int right = j < arr.length ? Math.abs(arr[j] - x) : Integer.MAX_VALUE;
            if(left <= right) --i;
            else ++j;
        }
        int l = i + 1, r = j - 1;
        for(int p = l; p <= r; ++p)
            res.add(arr[p]);
        return res;
    }
}
```

## 2.数学应用

```java
// 两数相除
class Solution {
    public int divide(int dividend, int divisor) {
        if(dividend == Integer.MIN_VALUE && divisor == -1)
            return Integer.MAX_VALUE;
        int sign = (divisor > 0) ^ (dividend > 0) ? -1 : 1;
        // 考虑极端情况下溢出，绝对值的除数、被除数与商都初始化为long型
        long absDividend = Math.abs((long)dividend);
        long absDivisor = Math.abs((long)divisor);
        long quotient = 0;
        while(absDividend >= absDivisor) {
            long count = 1;
            long sum = absDivisor;
            while((sum << 1) < absDividend) {
                count <<= 1;
                sum <<= 1;
            }
            quotient += count;
            absDividend -= sum;
        }
        return (int) (sign * quotient);
    }
}

// 求次幂Pow(x, n)
class Solution {
    public double myPow(double x, int n) {
        if(n == 0) return 1.0;
        if(n == 1) return x * 1.0;

        // 将n变为正数，便于递归
        if(n < 0) {
            if(n == Integer.MIN_VALUE) {
                x = 1 / x;
                x = x * x;
                n = Integer.MAX_VALUE;
            } else {
                n *= -1;
                x = 1 / x;
            }
        }
        return (n & 1) == 1 ? x * myPow(x * x, n >> 1) : myPow(x * x, n >> 1);
    }
}

// 求开方
class Solution {
    public int mySqrt(int x) {
        if(x <= 1) return x;

        int l = 0, r = x;
        while(true) {
            int m = l + ((r - l) >> 1);
            if(x / m < m)
                r = m - 1;
            else if(x / (m + 1) < (m + 1))
                return m;
            else
                l = m + 1;
        }
    }
}
```

# 二、前缀和

## 1.子数组和

```java
// 和为k的子数组
class Solution {
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0; // sum表示从nums[0]累加到nums[i]的累加和
        for(int i = 0; i < nums.length; ++i) {
            sum += nums[i];
            // 如果sum[i] - sum[j] == k，说明(j,i]之间的连续子数组累加和为k
            if(map.containsKey(sum - k))
                count += map.get(sum - k);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return count;
    }
}

// 连续的子数组和：数组大小至少为2且总和为k的倍数
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        if(nums == null || nums.length < 2) return false;
        int[] sum = new int[nums.length];
        sum[0] = nums[0];
        for(int i = 1; i < nums.length; ++i) {
            sum[i] = sum[i - 1] + nums[i];
        }
        for(int i = 0; i < nums.length; ++i) {
            for(int j = i + 1; j < nums.length; ++j) {
                int s = sum[j] - sum[i] + nums[i];
                if(s == 0 || s == k || (k != 0 && s >= k && s % k == 0))
                    return true;
            }
        }
        return false;
    }
}
```

## 2.二叉树路径总和

```java
// 路径方向向下
class Solution {
    public int pathSum(TreeNode root, int sum) {
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        return pathSum(root, sum, map, 0);
    }

    private int pathSum(TreeNode root, int sum, HashMap<Integer, Integer> map, int curSum) {
        if(root == null) return 0;
        curSum += root.val;
        int res = map.getOrDefault(curSum - sum, 0);
        map.put(curSum, map.getOrDefault(curSum, 0) + 1);
        res += pathSum(root.left, sum, map, curSum) + pathSum(root.right, sum, map, curSum);
        map.put(curSum, map.get(curSum) - 1);
        return res;
    }
}
```

## 3.优美数组：k个奇数的子数组

```java
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
        if(nums == null) return 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0, res = 0;
        for(int num : nums) {
            if(num % 2 != 0)
                ++count;
            map.put(count, map.getOrDefault(count, 0) + 1);
            res += map.getOrDefault(count - k, 0);
        }
        return res;
    }
}
```

# 三、双指针

## 1.双指针

```java
// 长度最小的子数组
class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        if(nums.length == 0) return 0;
        int sum = 0;
        int res = Integer.MAX_VALUE;
        for(int l = 0, r = 0; r < nums.length; ++r) {
            sum += nums[r];
            // 滑动窗口：双指针
            while(sum >= s) {
                res = Math.min(res, r - l + 1);
                sum -= nums[l++];
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }
}

// 三数之和
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; ++i) {
            if(nums[i] > 0) break;
            else if(i > 0 && nums[i] == nums[i - 1]) continue;
            int target = 0 - nums[i];
            for(int l = i + 1, r = nums.length - 1; l < r; ) {
                if(nums[l] + nums[r] == target) {
                    res.add(Arrays.asList(nums[i], nums[l++], nums[r--]));
                    while(l < r && nums[l] == nums[l - 1]) ++l;
                    while(l < r && nums[r] == nums[r + 1]) --r;
                } else if(nums[l] + nums[r] < target) ++l;
                else --r;
            }
        }
        return res;
    }
}

// 最接近的三数之和
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int sum = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; ++i) {
            for(int l = i + 1, r = nums.length - 1; l < r; ) {
                int newSum = nums[i] + nums[l] + nums[r];
                if(newSum == target) return target;
                else if(newSum < target) {
                    while(l < r && nums[l] == nums[l + 1]) ++l;
                    ++l;
                } else {
                    while(l < r && nums[r] == nums[r - 1]) --r;
                    --r;
                }
                if(Math.abs(target - sum) > Math.abs(target - newSum))
                    sum = newSum;
            }
        }
        return sum;
    }
}

// 四数之和
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length < 4) return res;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 3; ++i){
            // 提高计算速度的关键就在于下面两行
            // 如果从i开始连续4个最小的数字之和都大于target，说明现在不会再有4数之和等于target了
            // 如果i和后面最大的3个数加起来之和小于target，说明第i个数还不够大，需要直接增加第i个数
            if(nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target)
                break;
            if(nums[i] + nums[nums.length - 3] + nums[nums.length - 2] + nums[nums.length - 1] < target)
                continue;
            for(int j = i + 1; j < nums.length - 2; ++j){
                int targ = target - nums[i] - nums[j];
                for(int p = j + 1, q = nums.length - 1; p < q; ){
                    if(nums[p] + nums[q] == targ){
                        res.add(Arrays.asList(nums[i], nums[j], nums[p], nums[q]));
                        while(p < q && nums[p + 1] == nums[p])
                            ++p;
                        while(p < q && nums[q - 1] == nums[q])
                            --q;
                        ++p;
                        --q;
                    }
                    else if(nums[p] + nums[q] < targ)
                        ++p;
                    else
                        --q;
                }
                while(j < nums.length - 2 && nums[j + 1] == nums[j])
                    ++j;
            }
            while(i < nums.length - 3 && nums[i + 1] == nums[i])
                ++i;
        }
        return res;
    }
}
```

## 2.滑动窗

```java
// 大小为 K 且平均值大于等于阈值的子数组数目
class Solution {
    public int numOfSubarrays(int[] arr, int k, int threshold) {
        if(arr == null || arr.length < k)
            return 0;
        int target = k * threshold;
        int sum = 0;
        for(int i = 0; i < k; ++i)
            sum += arr[i];

        int res = sum >= target ? 1 : 0;
        for(int i = 0; i < arr.length - k; ++i) {
            sum += arr[i + k] - arr[i];
            if(sum >= target) ++res;
        }
        return res;
    }
}
```

## 3.数组就地修改

```java
// 删除排序数组中的重复项，使得最多出现2次
class Solution {
    public int removeDuplicates(int[] nums) {
        int i = 0;
        for(int num : nums)
            if(i < 2 || num > nums[i - 2])
                nums[i++] = num;
        return i;
    }
}
```

# 四、排序算法

## 1.快速选择

```java
// 数组中第k大的数
class Solution {
    public int findKthLargest(int[] nums, int k) {
        if(nums == null || nums.length < k)
            return -1;
        int left = 0, right = nums.length - 1;
        while(left < right) {
            int pivotIdx = partition(nums, left, right);
            if(pivotIdx == k - 1) break;
            if(pivotIdx < k - 1) left = pivotIdx + 1;
            else right = pivotIdx - 1;
        }
        return nums[k - 1];
    }

    private int partition(int[] nums, int left, int right) {
        int pivot = left + (int)(Math.random() * (right - left + 1));
        if(pivot != left)
            swap(nums, pivot, left);
        int pivotNum = nums[left];
        int l = left, r = right;
        while(l < r) {
            while(l < r && nums[r] < pivotNum)
                --r;
            while(l < r && nums[l] >= pivotNum)
                ++l;
            if(l != r) swap(nums, l, r);
        }
        swap(nums, l, left);
        return l;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

## 2.归并排序

```java
// 计算右侧小于当前元素的个数
class Solution {
    private int[] temp;
    private int[] index;
    private int[] counter;
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums == null || nums.length == 0)
            return res;
        int n = nums.length;
        
        temp = new int[n];
        index = new int[n];
        counter = new int[n];
        for(int i = 0; i < n; ++i)
            index[i] = i;
        mergeSort(nums, 0, n - 1);
        for(int count : counter)
            res.add(count);
        return res;
    }

    private void mergeSort(int[] nums, int start, int end) {
        if(start >= end)
            return;
        int mid = start + ((end - start) >> 1);
        mergeSort(nums, start, mid);
        mergeSort(nums, mid + 1, end);
        merge(nums, start, mid ,end);
    }

    private void merge(int[] nums, int start, int mid, int end) {
        int left = start, right = mid + 1, pos = start;
        // 合并的同时统计逆序对的数目，当且仅当左边的有序序列数字出列的时候才会计数
        // 因为左边数字出列说明左边数字当前最小，可以统计出比该数字小的右侧数字的个数了
        while(left <= mid && right <= end) {
            if(nums[index[left]] > nums[index[right]]) {
                temp[pos++] = index[right++];
            } else {
                counter[index[left]] += right - 1 - mid;
                temp[pos++] = index[left++];
            }
        }
        while(left <= mid) {
            counter[index[left]] += end - mid;
            temp[pos++] = index[left++];
        }
        while(right <= end) {
            temp[pos++] = index[right++];
        }
        for(int i = start; i <= end; ++i)
            index[i] = temp[i];
    }
}

// 翻转对
class Solution {
    private int[] helper;
    public int reversePairs(int[] nums) {
        // 归并排序的思想：将数组分成left和right两边，第一个数字在left中第二个数字在right中
        // 当left与right分别是已经排好序的数组时，可以快速求出共有多少逆序对
        this.helper = new int[nums.length];
        return mergeSort(nums, 0, nums.length - 1);
    }

    private int mergeSort(int[] nums, int start, int end) {
        if(start >= end) return 0;


        int mid = start + ((end - start) >> 1);
        int count = mergeSort(nums, start, mid) + mergeSort(nums, mid + 1, end);
        for(int i = start, j = mid + 1; i <= mid; ++i) {
            while(j <= end && (nums[i] / 2.0 > nums[j])) ++j; // double型
            count += j - (mid + 1);
        }

        merge(nums, start, mid, end);
        return count;
    }

    private void merge(int[] nums, int start, int mid, int end) {
        for(int i = start; i <= end; ++i)
            helper[i] = nums[i];
        int i = start, j = mid + 1, p = start;
        while(i <= mid || j <= end) {
            if(i > mid || (j <= end && helper[i] > helper[j]))
                nums[p++] = helper[j++];
            else
                nums[p++] = helper[i++];
        }
    }
}
```

## 3.桶排序

```java
// 任务调度器
class Solution {
    public int leastInterval(char[] tasks, int n) {
        if(tasks == null) return 0;

        int[] counter = new int[26];
        for(char task : tasks)
            ++counter[task - 'A'];
        // 拥有任务最多的任务类型的任务数决定了桶的数量
        int bucketNum = 0;
        for(int i = 0; i < 26; ++i) {
            if(counter[i] != 0)
                bucketNum = Math.max(bucketNum, counter[i]);
        }
        // 任务最多的任务类型种数决定了最后一个桶的大小
        int maxTaskNum = 0;
        for(int i = 0; i < 26; ++i)
            if(counter[i] == bucketNum) ++maxTaskNum;
        
        int time1 = tasks.length; // 冷却时间短，不会有待命状态
        int time2 = (bucketNum - 1) * (n + 1) + maxTaskNum;
        return Math.max(time1, time2);
    }
}

// 最大间距
class Solution {
    public int maximumGap(int[] nums) {
        int n = nums.length;
        if(n < 2) return 0;
        // 计算nums数组中的最小值和最大值
        int min = nums[0], max = nums[0];
        for(int num : nums) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        // 共有n-1个桶，将n-2个数（也就是除去max和min）放置在这些桶里，至少会有1个空桶
        // 空桶保证了桶内数字的最大gap必然小于桶间的gap
        int bucketGap = (int)Math.ceil((double)(max - min) / (n - 1)); // 注意这里的double运算和int转换
        // 各个桶只需要记录桶内的最大数据与最小数据即可
        int[] bucketMin = new int[n - 1];
        int[] bucketMax = new int[n - 1];
        Arrays.fill(bucketMin, Integer.MAX_VALUE);
        Arrays.fill(bucketMax, Integer.MIN_VALUE);
        // 将数字装入桶中
        for(int num : nums) {
            if(num == min || num == max) // 最小值与最大值不加入桶中
                continue;
            int bucketIdx = (num - min) / bucketGap; // 数字num应装入第bucketIdx桶中
            bucketMin[bucketIdx] = Math.min(bucketMin[bucketIdx], num);
            bucketMax[bucketIdx] = Math.max(bucketMax[bucketIdx], num);
        }
        // 求取最大gap
        int maxGap = Integer.MIN_VALUE; // min也要参与计算
        int preMax = min; // 前一个桶的最大值，也是bucketMin[i]的前一个数字
        for(int i = 0; i < n - 1; ++i) {
            if(bucketMax[i] == Integer.MIN_VALUE) // 说明桶是空的
                continue;
            maxGap = Math.max(maxGap, bucketMin[i] - preMax);
            preMax = bucketMax[i];
        }
        maxGap = Math.max(maxGap, max - preMax);
        return maxGap;
    }
}

// 前k个高频元素
class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        // 桶排序：将出现频率相同的数字放入一个桶中
        List<Integer> res = new ArrayList<>();
        HashMap<Integer, Integer> frequencyMap = new HashMap<>();
        for(int num : nums)
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        
        List<Integer>[] bucket = new ArrayList[nums.length + 1];
        // 把数字放入桶中
        for(Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            int freq = entry.getValue();
            if(bucket[freq] == null) bucket[freq] = new ArrayList<>();
            bucket[freq].add(entry.getKey());
        }
        // 统计频率最高的k个数字
        for(int freq = nums.length; freq >= 1 && k > 0; --freq) {
            if(bucket[freq] != null) {
                for(int i = 0; i < bucket[freq].size() && k > 0; ++i) {
                    res.add(bucket[freq].get(i));
                    --k;
                }
            }
        }
        return res;
    }
}
```

## 4.自定义排序

```java
// 合并区间
class Solution {
    public int[][] merge(int[][] intervals) {
        if(intervals.length <= 1) return intervals;
        Arrays.sort(intervals, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        List<int[]> res = new ArrayList<>();
        int[] newInterval = intervals[0];
        res.add(newInterval);
        for(int[] interval : intervals) {
            if(newInterval[1] >= interval[0])
                newInterval[1] = Math.max(newInterval[1], interval[1]);
            else {
                newInterval = interval;
                res.add(newInterval);
            }
        }
        return res.toArray(new int[res.size()][]);
    }
}

// 插入区间
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        // intervals最初已经是升序排列
        List<int[]> res = new ArrayList<>();
        int i = 0, n = intervals.length;
        // 先将完全小于新序列的序列放到res中
        while(i < n && intervals[i][1] < newInterval[0]) {
            res.add(new int[]{intervals[i][0], intervals[i][1]});
            ++i;
        }
        
        // 将有重叠的序列合并后放到res中
        // 合并后的新序列左边界必然小于下一个序列的右边界
        // 若新序列的右边界大于等于下一个序列的左边界，则表示重合了
        while(i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(intervals[i][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[i][1], newInterval[1]);
            ++i;
        }
        res.add(new int[]{newInterval[0], newInterval[1]});
        
        // 将完全大于合并后的新序列的序列插入res中
        while(i < n) {
            res.add(new int[]{intervals[i][0], intervals[i][1]});
            ++i;
        }

        int[][] ret = new int[res.size()][2];
        for(int j = 0; j < res.size(); ++j)
            ret[j] = res.get(j);
        
        return ret;
    }
}
```

## 5.其他

```java
// 颜色分类——就地一趟扫描排序
class Solution {
    public void sortColors(int[] nums) {
        int idx0 = 0, idx1 = 0, idx2 = 0;
        for(int num : nums) {
            if(num == 0) {
                nums[idx2++] = 2;
                nums[idx1++] = 1;
                nums[idx0++] = 0;
            } else if(num == 1) {
                nums[idx2++] = 2;
                nums[idx1++] = 1;
            } else nums[idx2++] = 2;
        }
    }
}

// 摆动排序
class Solution {
    public void wiggleSort(int[] nums) {
        if(nums == null || nums.length < 2)
            return;
        int n = nums.length;
        int i = 0, s = 0, l = 0, mid = (n >> 1) + n % 2;
        Arrays.sort(nums);
        int[] small = Arrays.copyOfRange(nums, 0, mid);
        int[] large = Arrays.copyOfRange(nums, mid, n);
        // s与l都是从后向前
        i = 0; s = small.length - 1; l = large.length - 1;
        while(i < n) {
            if(i % 2 == 0)
                nums[i++] = small[s--];
            else
                nums[i++] = large[l--];
        }
    }
}

// 两个数组的交集
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        if(nums1 == null || nums2 == null)
            return new int[]{};
        if(nums1.length < nums2.length)
            return intersect(nums2, nums1);

        Arrays.sort(nums1);
        Arrays.sort(nums2);
        List<Integer> list = new ArrayList<>();
        for(int i = 0, j = 0; i < nums1.length && j < nums2.length; ) {
            if(nums1[i] == nums2[j]) {
                list.add(nums1[i++]);
                ++j;
            } else if(nums1[i] > nums2[j]) {
                while(j < nums2.length && nums2[j] < nums1[i]) ++j;
            } else {
                while(i < nums1.length && nums1[i] < nums2[j]) ++i;
            }
        }
        int[] res = new int[list.size()];
        for(int i = 0; i < list.size(); ++i)
            res[i] = list.get(i);
        return res;
    }
}
```

# 五、摩尔投票法

```java
// 超过2/n的多数元素
class Solution {
    public int majorityElement(int[] nums) {
        // 摩尔投票法
        int cnt = 1, group = nums[0]; // group表示当前队伍，cnt表示当前队伍的人数
        for(int i = 1; i < nums.length; ++i) {
            if(cnt == 0) { // 当前队伍人数为0，更新队伍
                cnt = 1;
                group = nums[i];
                continue;
            }
            if(group == nums[i]) // 遇到相同队伍的人，人数+1
                ++cnt;
            else
                --cnt;
        }
        return group;
    }
}

// 超过1/3的众数
class Solution {
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums == null || nums.length == 0) return res;

        int cand1 = nums[0], count1 = 0;
        int cand2 = nums[0], count2 = 0;
        for(int num : nums) {
            if(num == cand1) {
                ++count1;
                continue;
            }
            if(num == cand2) {
                ++count2;
                continue;
            }
            if(count1 == 0) {
                cand1 = num;
                count1 = 1;
                continue;
            }
            if(count2 == 0) {
                cand2 = num;
                count2 = 1;
                continue;
            }

            --count1;
            --count2;
        }

        count1 = 0;
        count2 = 0;
        for(int num : nums) {
            if(num == cand1) ++count1;
            else if(num == cand2) ++count2;
        }
        if(count1 > nums.length / 3) res.add(cand1);
        if(count2 > nums.length / 3) res.add(cand2);

        return res;
    }
}
```

# 六、循环数组

```java
// 加油站
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if(gas == null || gas.length == 0) return -1;
        int n = gas.length;
        for(int start = 0; start < n; ) {
            int tank = 0, step = 1;
            for( ; step <= n; ++step) {
                int curStation = (start + step - 1) % n;
                if(tank + gas[curStation] < cost[curStation])
                    break;
                tank += gas[curStation] - cost[curStation];
            }
            if(step == n + 1) return start;
            start += step;
        }
        return -1;
    }
}

// 旋转数组
// 解法1：就地改值
class Solution {
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k = k % n;
        if(k == 0) return;
        for(int start = 0, count = 0; start < n && count < n; ++start) {
            for(int end = (start + k) % n, preVal = nums[start]; count < n; end = (end + k) % n) {
                int nextPre = nums[end];
                nums[end] = preVal;
                preVal = nextPre;
                ++count;
                if(end == start) break;
            }
        }
    }
}
// 解法2：参考翻转字符串
class Solution {
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int step = k % n;
        if(step == 0) return;
        reverse(nums, 0, n - 1); // 先全部逆转过来
        reverse(nums, 0, step - 1); // 再把前半部分逆转
        reverse(nums, step, n - 1); // 最后把后半部分逆转
    }
    
    private void reverse(int[] nums, int start, int end) {
        while(start < end) {
            int temp = nums[start];
            nums[start++] = nums[end];
            nums[end--] = temp;
        }
    }
}
// 解法3：循环交换
class Solution {
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int step = k % n;
        if(step == 0) return;
        int cnt = 0; // 统计已经换过位置的单词数
        for(int startIdx = 0; cnt < n; ++startIdx) {
            int curIdx = startIdx;
            int preVal = nums[startIdx];
            do {
                int nextIdx = (curIdx + step) % n;
                int temp = nums[nextIdx];
                nums[nextIdx] = preVal;
                preVal = temp;
                curIdx = nextIdx;
                ++cnt;
            } while(curIdx != startIdx); // 当前数字回到起点时说明转完一圈了
        }
    }
}
```

# 七、二维数组操作

```java
// 顺时针90度旋转图像
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for(int loop = 0; loop < (n >> 1); ++loop) {
            for(int i = loop; i < n - loop - 1; ++i) {
                swap(matrix, loop, i, i, n - loop - 1);
                swap(matrix, loop, i, n - loop - 1, n - i - 1);
                swap(matrix, loop, i, n - i - 1, loop);
            }
        }
    }

    private void swap(int[][] matrix, int x, int y, int i, int j) {
        int temp = matrix[x][y];
        matrix[x][y] = matrix[i][j];
        matrix[i][j] = temp;
    }
}

// 螺旋矩阵
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if(matrix.length == 0 || matrix[0].length == 0) return res;
        int left = 0, right = matrix[0].length - 1;
        int up = 0, down = matrix.length - 1;
        while(left <= right && up <= down) {
            // 上边界
            for(int i = left; i <= right; ++i)
                res.add(matrix[up][i]);
            if(++up > down) break;
            // 右边界
            for(int i = up; i <= down; ++i)
                res.add(matrix[i][right]);
            if(--right < left) break;
            // 下边界
            for(int i = right; i >= left; --i)
                res.add(matrix[down][i]);
            if(--down < up) break;
            // 左边界
            for(int i = down; i >= up; --i)
                res.add(matrix[i][left]);
            if(++left > right) break;
        }
        return res;
    }
}

// 矩阵置0
class Solution {
    public void setZeroes(int[][] matrix) {
        // 难点在于如果边遍历边置0，前面置0的会误导对后面的判断
        // 将(i,j)是否为0转换到首行首列标记
        boolean firstColHasZero = false;
        int m = matrix.length, n = matrix[0].length;
        
        for(int i = 0; i < m; ++i) {
            if(matrix[i][0] == 0) // 如果首列有0
                firstColHasZero = true;
            for(int j = 1; j < n; ++j) {
                // 对该坐标的首行和首列做标记
                if(matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        
        // 根据标记对各行各列置0
        // 必须从最后一行最后一列倒着置0到第一行第一列
        for(int i = m - 1; i >= 0; --i) {
            for(int j = n - 1; j >= 1; --j) {
                if(matrix[i][0] == 0 || matrix[0][j] == 0)
                    matrix[i][j] = 0;
            }
            if(firstColHasZero)
                matrix[i][0] = 0;
        }
    }
}
```

# 八、贪心算法

```java
// 跳跃游戏
class Solution {
    public int jump(int[] nums) {
        int n = nums.length, res = 0;
        
        for(int cur = 0; cur < n - 1; ) {
            int maxdist = 0, maxIndex = 0;
            for(int next = cur + nums[cur]; next > cur; --next) {
                if(next >= n - 1)
                    return res + 1;
                if(next - cur + nums[next] > maxdist) {
                    maxIndex = next;
                    maxdist = next - cur + nums[next];
                }
            }
            cur = maxIndex;
            ++res;
        }
        return res;
    }
}

// 跳跃游戏——判断能否到达终点
class Solution {
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int canReachIdx = 0;
        for(int i = 0; i < n && i <= canReachIdx; ++i)
            canReachIdx = Math.max(canReachIdx, i + nums[i]);
        
        return canReachIdx >= n - 1;
    }
}
```