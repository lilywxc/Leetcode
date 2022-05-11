# Leetcode
* [Two Pointers](#Two-Pointers)
    * [167. Two Sum II](#167-Two-Sum-II)
    * [633. Sum of Square Numbers](#633-Sum-of-Square-Numbers)
    * [345. Reverse Vowels of a String](#345-Reverse-Vowels-of-a-String)
    * [680. Valid Palindrome II](#680-Valid-Palindrome-II)
    * [88. Merge Sorted Array](#88-Merge-Sorted-Array)
    * [141. Linked List Cycle](#141-Linked-List-Cycle)
    * [524. Longest Word in Dictionary through Deleting](#524-Longest-Word-in-Dictionary-through-Deleting)
* [Sorting](#Sorting)
    * [215. Kth Largest Element in an Array](#215-Kth-Largest-Element-in-an-Array)
    * [347. Top K Frequent Elements](#347-Top-K-Frequent-Elements)
    * [451. Sort Characters By Frequency](#451-Sort-Characters-By-Frequency)
    * [75. Sort Colors](#75-Sort-Colors)
* [Greedy](#Greedy)
    * [455. Assign Cookies](#455-Assign-Cookies)
    * [435. Non overlapping Intervals](#435-Non-overlapping-Intervals)
    * [452. Minimum Number of Arrows to Burst Balloons](#452-Minimum-Number-of-Arrows-to-Burst-Balloons)
    * [406. Queue Reconstruction by Height](#406-Queue-Reconstruction-by-Height)
    * [121. Best Time to Buy and Sell Stock](#121-Best-Time-to-Buy-and-Sell-Stock)
    * [122. Best Time to Buy and Sell Stock II](#122-Best-Time-to-Buy-and-Sell-Stock-II)
    * [392. Is Subsequence](#392-Is-Subsequence)
    * [665. Non decreasing Array](#665-Non-decreasing-Array)
    * [53. Maximum Subarray](#53-Maximum-Subarray)
    * [763. Partition Labels](#763-Partition-Labels)
* [Divide and Conquer](#Divide-and-Conquer)
    * [241. Different Ways to Add Parentheses](#241-Different-Ways-to-Add-Parentheses)
    * [96. Unique Binary Search Trees]([#96-Unique-Binary-Search-Trees)
    * [95. Unique Binary Search Trees II](#95-Unique-Binary-Search-Trees-II)
* [Binary Search](#Binary-Search)
    * [69. Sqrt x](#69-Sqrt-x)
    * [744. Find Smallest Letter Greater Than Target](#744-Find-Smallest-Letter-Greater-Than-Target)
    * [540. Single Element in a Sorted Array](#540-Single-Element-in-a-Sorted-Array)
    * [278. First Bad Version](#278-First-Bad-Version)
    * [153. Find Minimum in Rotated Sorted Array](#153-Find-Minimum-in-Rotated-Sorted-Array)
    * [34. Find First and Last Position of Element in Sorted Array](#34-Find-First-and-Last-Position-of-Element-in-Sorted-Array)
* [Search](#Search)
	* [BFS](#BFS)
		* [1091. Shortest Path in Binary Matrix](#1091-Shortest-Path-in-Binary-Matrix)
		* [279. Perfect Squares](#279-Perfect-Squares) 
		* [127. Word Ladder](#127-Word-Ladder)
	* [DFS](#DFS)
		* [695. Max Area of Island](#695-Max-Area-of-Island)
		* [200. Number of Islands](#200-Number-of-Islands)
		* [547. Number of Provinces](#547-Number-of-Provinces)
		* [130. Surrounded Regions](#130-Surrounded-Regions)
		* [417. Pacific Atlantic Water Flow](#417-Pacific-Atlantic-Water-Flow)
	* [Backtracking](#Backtracking)
		* [17. Letter Combinations of a Phone Number](#17-Letter-Combinations-of-a-Phone-Number)


### Two Pointers
#### [167. Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)
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
```

```python
# Solution 2: dictionary - O(n) time and O(n) space
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        dict = {}
        for idx, num in enumerate(numbers):
            if num in dict:
                return [dict[num] + 1, idx + 1]
            else:
                dict[target - num] = idx   
```

```python          
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

#### [633. Sum of Square Numbers](https://leetcode.com/problems/sum-of-square-numbers/)
```python
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

```python     
# Solution 2: hashset - O(sqrt(n)) time and O(sqrt(n)) space
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        sqrtSet = set()
        
        for x in range(int(sqrt(c)) + 1):
            sqrtSet.add(x**2)
            
        for x in sqrtSet:
            if c - x in sqrtSet:
                return True
        
        return False
```

#### [345. Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/) 
```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        if s == None or len(s) == 0: 
            return s
        
        s = list(s)
        vows = set('aeiouAEIOU')
        
        l, r = 0, len(s) - 1
        while l < r:
            while l <= r and s[l] not in vows: 
                l += 1
            while l <= r and s[r] not in vows: 
                r -= 1
            if l > r: 
                break
                
            s[l], s[r] = s[r], s[l]
            l, r = l + 1, r - 1
            
        return ''.join(s)
```

#### [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)
```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        
        while l < r:
            if s[l] != s[r]:
                opt1 = s[l:r]
                opt2 = s[l + 1:r + 1]
                
                return opt1 == opt1[::-1] or opt2 == opt2[::-1]
            
            l, r = l + 1, r - 1
            
        return True
```        

#### [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        p1, p2 = m - 1, n - 1
        
        for p in range(n + m - 1, -1, -1):
            if p2 < 0:
                break
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                    nums1[p] = nums1[p1]
                    p1 -= 1        
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
```

#### [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None:
            return False
        
        slow = head
        fast = head.next
        
        while slow is not fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next
        
        return True
```
Time complexity : O(n), where n is the total number of nodes in the linked list. 
Consider the following two cases separately.
1. List has no cycle:
The fast pointer reaches the end first and the run time depends on the list's length, which is O(n).

2. List has a cycle:
Consider breaking down the movement of the slow pointer into two steps, the non-cyclic part (N nodes) and the cyclic part (K nodes):
- The slow pointer takes "N" steps to enter the cycle. At this point, the fast pointer has already entered the cycle. Run time = N
- Both pointers are now in the cycle. Consider two runners running in a cycle - the fast runner moves 2 steps while the slow runner moves 1 steps at a time. To catch up with the slow runner, the number of steps that fast runner needs is (distance between the 2 runners)/(difference of speed). As the distance is at most "K" and the speed difference is 1, we conclude that run time = K

Therefore, the worst case time complexity is O(N+K), which is O(n).

#### [524. Longest Word in Dictionary through Deleting](https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/)
```python
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        bestMatch = ""
        
        for word in dictionary:
            i = 0
            for char in s:
                if i < len(word) and char == word[i]:
                    i += 1
            
            if i == len(word):
                if (len(word) > len(bestMatch) or len(word) == len(bestMatch) and word < bestMatch):
                    bestMatch = word
                    
        return bestMatch
```

### Sorting

#### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
top K elements method could be solved using heap or quick select
- heap: we use min-heap of size k to store the largest k elements (since the smallest element is popped out, what're left in the heap at the end are the largest k elements)
- quick select: based on quicksort using partition
```python
# Solution 1: sorting - O(nlogn) time and O(1) space
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return sorted(nums, reverse=True)[k-1]
```

```python
# Solution 2: heapsort - O(k + (n-k)logk) time and O(k) space
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for n in nums:
            if len(heap) == k:
                heapq.heappushpop(heap, n)
            else:
                heapq.heappush(heap, n)
            
        return heapq.heappop(heap)
        # return heapq.nlargest(k, nums)[-1]
```

```python        
# Solution 3: quickselect - O(n) time and O(1) space
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        l, r = 0, len(nums) - 1
        
        while True:
            idx = self.partition(nums, l, r)
            
            if idx > k - 1:
                r = idx - 1
            elif idx < k - 1:
                l = idx + 1
            else:
                break
        
        return nums[idx]
        
    def partition(self, arr, start, end):
        pivot = arr[start]
        l = start + 1
        r = end
        
        while l <= r:
            if arr[l] < pivot and arr[r] > pivot:
                arr[l], arr[r] = arr[r], arr[l]
                l += 1
                r -= 1
            if arr[l] >= pivot:
                l += 1
            if arr[r] <= pivot:
                r -= 1
            
        arr[start], arr[r] = arr[r], arr[start] 
        return r # the position of the pivot
        
        # r will stop at a place where all numbers on the right are smaller than pivot, 
        # and l will stop at a place where all numbers on the left are bigger than pivot, including r (_ _ _ r l _ _). 
        # Thus swaping pivot and r gives the final list

```

#### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
```python
# Solution 1: bucket sort - O(n) time and O(n) space
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        freqMap = Counter(nums)
        
        max_freq = max(freqMap.values())
        buckets = [[] for _ in range(max_freq + 1)]  
        
        for num, freq in freqMap.items():
            buckets[freq].append(num)
        
        res = []
        for bucket in buckets[::-1]:
            if bucket:
                for num in bucket:
                    res.append(num)
                    
        return res[:k]
	
# Counter() is collections, and it has methods keys(), values(), items(), but it's not subscriptable, i.e., we cannot do Counter()[3]
# * Counter(nums).most_common(k) will return the result directly

# different ways to create frequency map
# 1. 
# freqMap = {}
# for num in nums:
# 	if num in d:
# 		freqMap[num] += 1
# 	else:
# 		freqMap[num] = 1
# 2.
# freqMap = collections.defaultdict(int)
# for num in nums:
#     freqMap[num] += 1
```
    
```python
# Solution 2: heap sort - O(nlogk) and O(n) space
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        freqMap = Counter(nums)

        heap = []
        for num, count in freqMap.items(): # O(nlog(k))
            if len(heap) == k:
                heappushpop(heap, (count, num))
            else:
                heappush(heap, (count, num))

        res = []
        while heap: # O(klogk)
            count, num = heappop(heap) 
            print(num, count)
            res.append(num)

        return res
```

#### [451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/solution/)
```python
class Solution:
    def frequencySort(self, s: str) -> str:
        freqMap = collections.Counter(s)
        max_freq = max(freqMap.values())
        
        buckets = [[] for _ in range(max_freq + 1)]
        
        for num, count in freqMap.items():
            buckets[count].append(num)
            
        stringBuilder = []
        for count in range(len(buckets) - 1, 0, -1):
            for char in buckets[count]:
                stringBuilder.append(char * count)
    
        return "".join(stringBuilder)
```

#### [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        p0, p2 = 0, len(nums) - 1 
        curr = 0
        
        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                curr += 1
                p0 += 1
            elif nums[curr] == 2:
                nums[p2], nums[curr] = nums[curr], nums[p2]
                p2 -= 1
            else:
                curr += 1
```

### Greedy
#### [455. Assign Cookies](https://leetcode.com/problems/assign-cookies/)
```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        
        child, cookie = 0, 0
        while child < len(g) and cookie < len(s):
            if g[child] <= s[cookie]:
                child += 1
                cookie += 1
            else:
                cookie += 1
                
        return child
```

#### [435. Non overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
for every selection, the end point is the most important - the smallest end point we choice, the more space left for the following intervals, and more intervals could be selected. We want to sort the intervals based on end point, and select the interval that has overlap with previous one each time.
```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        res = 0
        prevEnd = float('-inf')
        
        intervals.sort(key = lambda x: x[1]) # sort based on end point
        
        for intv in intervals:
            if intv[0] >= prevEnd: # non-overlap
                prevEnd = intv[1]
            else:
                res += 1  
        return res
```

#### [452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/)
```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        res = 0
        prevEnd = float('-inf')
        
        points.sort(key = lambda x: x[1])
        
        for intv in points:
            if intv[0] > prevEnd:
                res += 1
                prevEnd = intv[1]
                
        return res
```

#### [406. Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/description/)
Consider a queue with two 7-height people and one 6-height person. First, pick out tallest group of people (7-height) and sort them based on k. Since there's no other groups of people taller than them, each guy's index will be just as same as his k value.Now it's time to find a place for the guy of height 6. Since he is "invisible" for the 7-height guys, he could take whatever place without disturbing 7-height guys order. However, for him the others are visible, and hence he should take the position equal to his k-value, in order to have his proper place.

sorted List: [[7, 0], [7, 1], [6, 1], [5, 0], [5, 2], [4, 4]] <br />
[[7, 0]] <br />
[[7, 0], [7, 1]] <br />
[[7, 0], [6, 1], [7, 1]] <br />
[[5, 0], [7, 0], [6, 1], [7, 1]] <br />
[[5, 0], [7, 0], [5, 2], [6, 1], [7, 1]] <br />
[[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]] <br />

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key = lambda x: (-x[0], x[1]))
        output = []
        
        for p in people:
            output.insert(p[1], p)
            
        return output
```

#### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minPrice = float('inf') # min price in "previous days"
        maxProfit = 0
        
        for price in prices:
            minPrice = min(price, minPrice)
            maxProfit = max(price - minPrice, maxProfit)
            
        return maxProfit
# the brute force will be having a nested for loops in O(n^2)
```

#### [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]
                
        return profit
```

#### [392. Is Subsequence](https://leetcode.com/problems/is-subsequence/description/)
```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        for c in s:
            i = t.find(c)
            if i == -1:
                return False
            else:
                t = t[i + 1:]
        return True
# this problem can be solved easily by two pointers as well
```

#### [665. Non decreasing Array](https://leetcode.com/problems/non-decreasing-array/description/)
```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        violated = False

        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                if violated:
                    return False
                
                violated = True
                
                # when violation happens, nums[i - 1] > nums[i] 
                # we want to keep numbers as small as possible,
                # so that we have more flexibility with numbers later
                if i < 2 or nums[i - 2] <= nums[i]:
                    nums[i - 1] = nums[i] 
                else:
                    nums[i] = nums[i - 1]
                
        return True
```

#### [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/)
```python
# Solution 1: Greedy
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
            else:
                continue
                
        return max(nums)

# ex.    [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# nums = [-2, 1, -2, 4,  3, 5, 6,  1, 5]
```
```python
# Solution 2: devide and conquer
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        current_subarray = max_subarray = nums[0]
        
        for num in nums[1:]:
            # If current_subarray is negative, throw it away. Otherwise, keep adding to it.
            current_subarray = max(num, current_subarray + num)
            max_subarray = max(max_subarray, current_subarray)
        
        return max_subarray
```

#### [763. Partition Labels](https://leetcode.com/problems/partition-labels/description/)
```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # ex. "abccaddbeffe"
        
        last = {c: i for i, c in enumerate(s)}
        
        anchor = 0
        p = 0
        res = []
        for i, c in enumerate(s):
            p = max(p, last[c])
            if i == p:
                res.append(i - anchor + 1)
                anchor = i + 1
                
        return res
```
### Divide and Conquer
#### [241. Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)
- The problem becomes easier when we think about these expressions as expression trees.
- We can traverse over the experssion and whenever we encounter an operator, we recursively divide the expression into left and right part and evaluate them seperately until we reach a situation where our expression is purely a number and in this case we can simply return that number.
- Since there can be multiple ways to evaluate an expression (depending on which operator you take first) we will get a list of reults from left and the right part.
- Now that we have all the possible results from the left and the right part, we can use them to find out all the possible results for the current operator.
- note, we create a memo to get answers for repeated calculation
<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/241.%20Different%20Ways%20to%20Add%20Parentheses.png" width="700">

ex. "2*3-4*5" <br />
[4] * [5] <br />
res: 4*5 [20] <br />
[3] - [20] <br />
res: 3-4*5 [-17] <br />
[3] - [4] <br />
res: 3-4 [-1] <br />
[-1] * [5] <br />
res: 3-4*5 [-17, -5] <br />
[2] * [-17, -5] <br />
res: 2*3-4*5 [-34] <br />
res: 2*3-4*5 [-34, -10] <br />
[2] * [3] <br />
res: 2*3 [6] <br />
[6] - [20] <br />
res: 2*3-4*5 [-34, -10, -14] <br />
[2] * [-1] <br />
res: 2*3-4 [-2] <br />
[6] - [4] <br />
res: 2*3-4 [-2, 2] <br />
[-2, 2] * [5] <br />
res: 2*3-4*5 [-34, -10, -14, -10] <br />
res: 2*3-4*5 [-34, -10, -14, -10, 10] <br />
```python
class Solution:
    def diffWaysToCompute(self, expression: str, memo = {}) -> List[int]:
     
        if expression.isdigit():
            return [int(expression)]
        
        if expression in memo:
            return memo[expression]
        
        res = []
        for i, c in enumerate(expression):
            if c in '*+-':
                left = self.diffWaysToCompute(expression[:i])
                right = self.diffWaysToCompute(expression[i+1:])
                
                # print(left, c, right)
                for x in left:
                    for y in right:
                        res.append(self.compute(x, y, c))
                        # print('res:', expression, res)
       
        memo[expression] = res
        return res
                
    def compute(self, x, y, op):
        if op == "+":
            return x + y
        elif op == "-":
            return x - y
        else:
            return x * y
```

#### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)
```python
class Solution:
    def numTrees(self, num: int) -> int:
        G = [0]*(num + 1)
        G[0], G[1] = 1, 1 
        
        for n in range(2, num + 1):
            for i in range(1, n + 1):
                G[n] += G[i - 1]*G[n - i]
                
        return G[num]
```

#### [95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        
        def generateTrees(start, end):
            if start > end:
                return [None, ]
            
            res = []
            for i in range(start, end + 1):
                left = generateTrees(start, i - 1)
                right = generateTrees(i + 1, end)
                
                for l in left:
                    for r in right:
                        current = TreeNode(i)
                        current.left = l
                        current.right = r
                        res.append(current)
                        
            return res
        
        return generateTrees(1, n) if n else []
```

### Binary Search
ordinary/original binary search
```python
def binarySearch(nums, key):
   l = 0
   r = len(nums) - 1
   
   while l <= r:
   	
	m = l + (r - l) // 2
	
	if nums[m] == key:
	    return m
	elif nums[m] > key:
            r = m - 1
        else:
            l = m + 1
  
    return -1
```

Variant example: find the leftmost match of a key in a non-decreasing array (has repeated numbers)
```python
def binarySearch(nums, key):
   l = 0
   r = len(nums) - 1
   
   while l < r:
   	
	m = l + (r - l) // 2
	
	if nums[m] >= key:
	    r = m
        else:
            l = m + 1
  
    return l
```
in this case where r is assigned m, we need while loop of l < r instead of l <= r to avoid possibly infinite loop

#### [69. Sqrt x](https://leetcode.com/problems/sqrtx/)
```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2:
            return x
            
        l = 2
        r = x // 2
        
        while l <= r:
            m = l + (r - l)//2
            square = m * m
            
            if square == x:
	    	return m
	    elif square > x:
                r = m - 1
            else:
                l = m + 1
            
        return r
```
We return r (the smaller) when we break out the while loop. Consider one step earlier, l = r = m: 
- if m^2 > key, r = m - 1. We know r now is smaller than x cuz that's a previous location of l, meaning the square < x for sure. And l should not be the answer because l = m and we have evaluated that m^2 > key. r is the answer
- If m^2 < key, l = m + 1. We know l now is bigger than x cuz that's a previous location of r, meaning the square > x for sure. And r should be the answer because l = m and we have evaluated that m^2 > key.

#### [744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)
```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        l = 0
        h = len(letters) - 1
        
        while l <= h:
            m = l + (h - l) // 2
            
            if letters[m] <= target:
                l = m + 1
            else:
                h = m - 1
        
        return letters[l] if l < len(letters) else letters[0]
```

#### [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/)
```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l = 0
        h = len(nums) - 1
        
        while l < h:
            m = l + (h - l) // 2 
            
            if m % 2 == 1: # make sure m is at even idex
                m -= 1
                
            if nums[m] == nums[m + 1]:
                l = m + 2
            else:
                h = m
                
        return nums[l]
	# when we break out the while loop, l = r, so both index l and r of nunms is correct
```

#### [278. First Bad Version](https://leetcode.com/problems/first-bad-version/)
```python
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        l = 0
        h = n
        
        while l < h:
            m = l + (h - l) // 2
            
            if isBadVersion(m):
                h = m
            else:
                l = m + 1
        
        return l
```

#### [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/submissions/)
the main idea for our checks is to converge the left and right bounds on the start
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l = 0
        h = len(nums) - 1
        
        while l < h:
            m = l + (h - l) // 2
            
            if nums[m] > nums[h]:
                l = m + 1
            else:
                h = m
                
        return nums[l]
```


#### [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        first = self.binarySearch(nums, target)
        
        if first == len(nums) or nums[first] != target:
            return [-1, -1]
        
        last = self.binarySearch(nums, target + 1) - 1
        
        return [first, last]
        
        
    def binarySearch(self, nums, key):
        l = 0
        r = len(nums)
        
        while l < r:
            m = l + (r - l) // 2

            if nums[m] >= key:
                r = m
            else:
                l = m + 1

        return l
```

### Search

#### BFS

#### [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        if grid[0][0] or grid[n-1][n-1]:
            return -1
        
        directions = [[1,0], [-1,0], [0,1], [0,-1], [-1,-1], [1,1], [1,-1], [-1,1]]
        
        q = collections.deque([(0,0,1)]) 
        while q:
            i, j, dist = q.popleft()
            if i == n - 1 and j == n - 1: 
                return dist
            
            for d1, d2 in directions:
                x, y = i + d1, j + d2
                if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                    grid[x][y] = 1
                    q.append((x, y, dist + 1))
                    
        return -1
```
note, if the input grid is not immutable, then we should have a set "seen" to store whether the cell has been visited or not, which takes O(N) - now the space is O(1) as we modify in place.

#### [279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)
<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/279.%20Perfect%20Squares.png" width="550">

```python
# Solution 1: BFS
class Solution:
    def numSquares(self, n: int) -> int:
        square_nums = [i**2 for i in range(1, int(math.sqrt(n))+1)]
        
        level = 0
        queue = {n} # we normally use queue in BFS, but we use set here to
                    # eliminate the redundancy of remainders within the same level
        
        while queue:
            level += 1
            
            next_queue = set()
            for remainder in queue:
                for square_num in square_nums:
                    if square_num == remainder:
                        return level
                    elif square_num >= remainder:
                        break
                    else:
                        next_queue.add(remainder - square_num)
        
            queue = next_queue
    
        return level
```

```python
# Solution 2: DP - O(N*√N) time and O(N) space
# numSquares(n) = min(numSquares(n-k) + 1) ∀k∈{square numbers}
class Solution:
    def numSquares(self, n: int) -> int:

        square_nums = [i**2 for i in range(0, int(math.sqrt(n))+1)]
        
        dp = [float('inf')] * (n+1)
        dp[0] = 0 # base case
        
        for i in range(1, n+1):
            for square in square_nums:
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i-square] + 1)
        
        return dp[-1]
```

#### [127. Word Ladder](https://leetcode.com/problems/word-ladder/)
```python
# BFS implementation using set
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if endWord not in wordSet: 
            return 0
            
        bq = {beginWord}
        eq = {endWord}
        dist = 1
        word_len = len(beginWord)
        
        while bq:
            dist += 1
            next_queue = set()
            wordSet -= bq # remove visited words
            
            for word in bq:
                for i in range(word_len):
                    for c in "abcdefghijklmnopqrstuvwxyz":
                        if c != word[i]:
                            new_word = word[:i] + c + word[i + 1:]
                             
                            if new_word in eq:                    
                                return dist
                            
                            if new_word in wordSet:
                                print(word, '->', new_word, dist)
                                next_queue.add(new_word)
                                wordSet.remove(new_word)
                                
            bq = next_queue
            if len(eq) < len(bq):
                bq, eq = eq, bq
                
        return 0
	
# BFS implementation using queue
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in set(wordList): 
            return 0
        
        # construct intermediate dict
        d = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                s = word[:i] + "_" + word[i+1:]
                d[s].append(word)
            
        queue, visited = deque([(beginWord, 1)]), set()
        while queue:
            word, steps = queue.popleft()
            if word not in visited:
                visited.add(word)
                
                if word == endWord:
                    return steps
                
                for i in range(len(word)):
                    s = word[:i] + "_" + word[i+1:]
                    
                    for adj in d[s]:
                        if adj not in visited:
                            queue.append((adj, steps + 1))
        return 0
```

### DFS

#### [695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/description/)
```python
# DFS recursive
class Solution:
    def maxAreaOfIsland(self, grid):
        nrow, ncol = len(grid), len(grid[0])

        def dfs(i, j):
            if 0 <= i < nrow and 0 <= j < ncol and grid[i][j]:
                grid[i][j] = 0 # mark visited cell in place, and get rid of seen set
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            
            return 0

        areas = [dfs(i, j) for i in range(nrow) for j in range(ncol) if grid[i][j]]
        
        return max(areas) if areas else 0
	
# DFS iterative
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        nrow, ncol = len(grid), len(grid[0])
        directions = [[1,0], [-1,0], [0,1], [0,-1]]
        seen = set()
        max_area = 0
        
        for r0 in range(nrow):
            for c0 in range(ncol):
                if grid[r0][c0] and (r0, c0) not in seen:
                    q = [(r0, c0)]
                    seen.add((r0, c0))
                    
                    cur_area = 0
                    while q:
                        r, c = q.pop()  # popleft() turns it to BFS 
                        cur_area += 1
                        
                        for d1, d2 in directions:
                            x, y = r + d1, c + d2
                            if 0 <= x < nrow and 0 <= y < ncol and grid[x][y] and (x, y) not in seen:
                                q.append((x, y))
                                seen.add((x, y))
                    
                    max_area = max(max_area, cur_area)
                    
        return max_area
```

#### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/description/)
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        nrow, ncol = len(grid), len(grid[0])

        def dfs(x, y):
            if 0 <= x < nrow and 0 <= y < ncol and grid[x][y] == "1":
                grid[x][y] = "0"
                for i, j in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    dfs(i, j)

        num = 0
        for x in range(nrow):
            for y in range(ncol):
                if grid[x][y] == "1":
                    dfs(x, y)
                    num += 1
                    
        return num
```

#### [547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/submissions/)
```python
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        N = len(M)
        seen = set()
        
        def dfs(node):
            for adj, isConnected in enumerate(M[node]):
                if isConnected and adj not in seen:
                    seen.add(adj)
                    dfs(adj)

        num = 0
        for i in range(N):
            if i not in seen:
                dfs(i)
                num += 1
                
        return num
```

#### [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/submissions/)
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:

        if not any(board): 
            return

        m, n = len(board), len(board[0])
        
        border = [idx for k in range(max(m, n)) for idx in ((0, k), (m-1, k), (k, 0), (k, n-1))]

        while border:
            i, j = border.pop()
            if 0 <= i < m and 0 <= j < n and board[i][j] == 'O':
                board[i][j] = 'B'
                border += [(i, j-1), (i, j+1), (i-1, j), (i+1, j)]

        for row in board:
            for i, c in enumerate(row):
                row[i] = 'O' if c == 'B' else 'X'
```

#### [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)
```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []

        p_visited = set()
        a_visited = set()
        rows, cols = len(heights), len(heights[0])

        def dfs(i, j, visited):
            visited.add((i, j))
            
            for (x, y) in ((i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)):
                if 0 <= x < rows and 0 <= y < cols and heights[x][y] >= heights[i][j] and (x, y) not in visited:
                    dfs(x, y, visited)

        for row in range(rows):
            dfs(row, 0, p_visited)
            dfs(row, cols - 1, a_visited)

        for col in range(cols):
            dfs(0, col, p_visited)
            dfs(rows - 1, col, a_visited)

        return list(p_visited.intersection(a_visited))
```

### Backtracking
#### [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/submissions/)
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0: 
            return []
        
        letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        
        def backtrack(index, path):
            if len(path) == len(digits):
                combinations.append("".join(path))
                return 
            
            possible_letters = letters[digits[index]]
            for letter in possible_letters:
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()

        combinations = []
        backtrack(0, [])
        return combinations
```
