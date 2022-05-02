# Leetcode
<!-- GFM-TOC -->
* [Leetcode 题解 - 双指针](#leetcode-题解---双指针)
    * [1. 有序数组的 Two Sum](#1-有序数组的-two-sum)
    * [2. 两数平方和](#2-两数平方和)
    * [3. 反转字符串中的元音字符](#3-反转字符串中的元音字符)
    * [4. 回文字符串](#4-回文字符串)
    * [5. 归并两个有序数组](#5-归并两个有序数组)
    * [6. 判断链表是否存在环](#6-判断链表是否存在环)
    * [7. 最长子序列](#7-最长子序列)
<!-- GFM-TOC -->


双指针主要用于遍历数组，两个指针指向不同的元素，从而协同完成任务。

## 1. 有序数组的 Two Sum


## Two Pointers
167. Two Sum II - Input Array Is Sorted
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
