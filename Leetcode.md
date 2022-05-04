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
Consider a queue with two 7-height people and one 6-height person. 
First, pick out tallest group of people (7-height) and sort them based on k. 
Since there's no other groups of people taller than them, each guy's index will be just as same as his k value.
Now it's time to find a place for the guy of height 6. 
Since he is "invisible" for the 7-height guys, he could take whatever place without disturbing 7-height guys order. 
However, for him the others are visible, and hence he should take the position equal to his k-value, in order to have his proper place.

sorted List: [[7, 0], [7, 1], [6, 1], [5, 0], [5, 2], [4, 4]]

[[7, 0]]

[[7, 0], [7, 1]]

[[7, 0], [6, 1], [7, 1]]

[[5, 0], [7, 0], [6, 1], [7, 1]]

[[5, 0], [7, 0], [5, 2], [6, 1], [7, 1]]

[[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key = lambda x: (-x[0], x[1]))
        output = []
        
        for p in people:
            output.insert(p[1], p)
            
        return output
```



