# Leetcode
* [Two Pointers](#Two-Pointers)
    * [167. Two Sum II - Input array is sorted](#167-Two-Sum-II)
    * [2. 两数平方和](#2-两数平方和)
    * [3. 反转字符串中的元音字符](#3-反转字符串中的元音字符)
    * [4. 回文字符串](#4-回文字符串)
    * [5. 归并两个有序数组](#5-归并两个有序数组)
    * [6. 判断链表是否存在环](#6-判断链表是否存在环)
    * [7. 最长子序列](#7-最长子序列)


### Two Pointers
#### 167. Two Sum II
```python
import numpy as np
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        n, r = divmod(abs(numerator), abs(denominator))
        sign = '-' if np.sign(numerator) * np.sign(denominator) < 0 else ''
        result = [sign + str(n), '.']
        stack = []
        while r not in stack:
            stack.append(r)
            n, r = divmod(r*10, abs(denominator))
            result.append(str(n))

        idx = stack.index(r)
        result.insert(idx+2, '(')
        result.append(')')
        return ''.join(result).replace('(0)', '').rstrip('.')
```
