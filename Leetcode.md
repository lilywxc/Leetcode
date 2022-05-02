# Leetcode
* [Two Pointers](#Two-Pointers)
    * [167. Two Sum II](#167-Two-Sum-II)
    * [633. Sum of Square Numbers](#633-Sum-of-Square-Numbers)
    * [3. 反转字符串中的元音字符](#3-反转字符串中的元音字符)
    * [4. 回文字符串](#4-回文字符串)
    * [5. 归并两个有序数组](#5-归并两个有序数组)
    * [6. 判断链表是否存在环](#6-判断链表是否存在环)
    * [7. 最长子序列](#7-最长子序列)


### Two Pointers
##### [167 Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)
```python
# Solution 1: two pointers - O(n) time and O(1) space
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers)-1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l+1, r+1]
            elif s > target:
                r -= 1
            else:
                l += 1

# Solution 2: dictionary - O(n) time and O(n) space
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        dict = {}
        for idx, num in enumerate(numbers):
            if num in dict:
                return [dict[num] + 1, idx + 1]
            else:
                dict[target - num] = idx
                
# Solution 3: binary search - O(nlogn) time and O(1) space
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        for i in range(len(numbers) - 1):
            tgt = target - numbers[i]
            l, r = i + 1, len(numbers) - 1
        
            while l <= r:
                mid = l + (r-l)//2

                if numbers[mid] == tgt:
                    return [i + 1, mid + 1]
                elif numbers[mid] < tgt:
                    l = mid + 1
                else:
                    r = mid - 1
```

[633 Sum of Square Numbers](https://leetcode.com/problems/sum-of-square-numbers/)
```
# Solution 1: two pointers - O(sqrt(n)) time and O(1) space
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        l, r = 0, int(sqrt(c))
        
        while l <= r:
            cur = l**2 + r**2
            
            if cur == c:
                return True
            elif cur < c:
                l += 1
            else:
                r -= 1
                
        return False
```



















































