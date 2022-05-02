# Leetcode
* [Two Pointers](#Two-Pointers)
    * [167. Two Sum II](#167-Two-Sum-II)
    * [2. 两数平方和](#2-两数平方和)
    * [3. 反转字符串中的元音字符](#3-反转字符串中的元音字符)
    * [4. 回文字符串](#4-回文字符串)
    * [5. 归并两个有序数组](#5-归并两个有序数组)
    * [6. 判断链表是否存在环](#6-判断链表是否存在环)
    * [7. 最长子序列](#7-最长子序列)


### Two Pointers
[167. Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)
```python
# Solution 1: two pointers - O(n) time and O(1) space
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers)-1
        while (l < r):
            s = numbers[l] + numbers[r]
            if (s == target):
                return [l+1, r+1]
            elif (s > target):
                r -= 1
            else:
                l += 1
```
