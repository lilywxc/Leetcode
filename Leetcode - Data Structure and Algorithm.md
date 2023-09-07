# Leetcode - Data Structure and Algorithm
* [Array](#Array)
* [Two Pointers](#Two-Pointers)
* [String](#String)
* [Matrix](#Matrix)
* [LinkedList](#LinkedList)
* [Tree](#Tree)
* [Stack and Queue](#Stack-and-Queue) 
* [HashMap](#HashMap)
* [Graph](#Graph)
* [Sorting](#Sorting)
* [Greedy](#Greedy)
* [Divide and Conquer](#Divide-and-Conquer)
* [Binary Search](#Binary-Search)
* [Search](#Search)
* [Dynamic Programming](#Dynamic-Programming)
* [Math](#Math)
* [Bit Computation](#Bit-Computation)


## Array 
* [76. Minimum Window Substring](#76-Minimum-Window-Substring)
* [209. Minimum Size Subarray Sum](#209-Minimum-Size-Subarray-Sum)
* [3. Longest Substring Without Repeating Characters](#3-Longest-Substring-Without-Repeating-Characters)
* [56. Merge Intervals](#56-Merge-Intervals)
* [283. Move Zeroes](#283-Move-Zeroes)
* [645. Set Mismatch](#645-Set-Mismatch)
* [41. First Missing Positive](#41-First-Missing-Positive)
* [287. Find the Duplicate Number](#287-Find-the-Duplicate-Number)
* [667. Beautiful Arrangement II](#667-Beautiful-Arrangement-II)
* [697. Degree of an Array](#697-Degree-of-an-Array)
* [565. Array Nesting](#565-Array-Nesting)
* [769. Max Chunks To Make Sorted](#769-Max-Chunks-To-Make-Sorted)
* [238. Product of Array Except Self](#238-Product-of-Array-Except-Self)
    

Things to look out during an interview:
1. Clarify if there are duplicate values in the array. Would the presence of duplicate values affect the answer? Does it make the question simpler or harder?
2. When using an index to iterate through array elements, be careful not to go out of bounds.
3. Be mindful about slicing or concatenating arrays in your code. Typically, slicing and concatenating arrays would take O(n) time. Use start and end indices to demarcate a subarray/range where possible.

Corner Caes:
1. Empty sequence
2. Sequence with 1 or 2 elements
3. Sequence with repeated elements

Techniques:
1. **Sliding window** (ex: [76](#76-Minimum-Window-Substring), [209](#209-Minimum-Size-Subarray-Sum), [3](#3-Longest-Substring-Without-Repeating-Characters)) <br />
the idea is to use a hashmap to check the validity of the window and have two pointers to adjust the window size 
	```
	1. Use two pointers: left and right to represent a window.
	2. Move right pointer to find a valid window.
	3. When a valid window is found, move left to contract and get a smaller window.
	```
2. **Two pointers** (ex. [75](#75-Sort-Colors), [88](#88-Merge-Sorted-Array))
3. **Traversing from the right**
4. **Sorting the array** (ex: [56](#56-Merge-Intervals), [435](#435-Non-overlapping-Intervals))
5. **Precomputation** (ex: [238](#238-Product-of-Array-Except-Self), [209](#209-Minimum-Size-Subarray-Sum)) <br />
for questions where summation or multiplication of a subarray is involved, pre-computation using hashing or a prefix/suffix sum/product might be useful
6. **Index has a hash key** (ex: [645](#645-Set-Mismatch), [287](#287-Find-the-Duplicate-Number), [41](#41-First-Missing-Positive)) <br />
This approach is usually used when interviewer asks for O(1) space. For example, if the array only has values from 1 to N, where N is the length of the array, negate the value at that index to indicate presence of that number


#### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
The idea is, we keep expanding the window by moving the right pointer. <br />
When the window has all the desired characters, we contract (if possible) and save the smallest window till now.
```python
# O(S+T) time and O(T) space
class Solution(object):
    def minWindow(self, s, t):
        if not t or not s:
            return ""

        dict_t = Counter(t)
        t_num = len(dict_t)
        l, r = 0, 0
        match_num = 0 # a char is considered 'matched' only if both char and count match
        window = defaultdict(int)

        res = (len(s) + 1, None, None) # (window length, left pointer, right pointer)

        while r < len(s):
            ch = s[r]
            
            if ch in dict_t:
                window[ch] += 1
                if window[ch] == dict_t[ch]:
                    match_num += 1

            while l <= r and match_num == t_num:
                ch = s[l]

                if r - l + 1 < res[0]:
                    res = (r - l + 1, l, r)

                if ch in dict_t:
                    window[ch] -= 1
                    if window[ch] < dict_t[ch]:
                        match_num -= 1

                l += 1    

            r += 1    
            
        return "" if res[0] == len(s) + 1 else s[res[1] : res[2] + 1]
```

#### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-window-substring/)
```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        total = 0
        min_length = len(nums) + 1
        
        l = 0
        for r, n in enumerate(nums):
            total += n
            while total >= s:
                min_length = min(min_length, r - l + 1)
                total -= nums[l]
                l += 1
                
        return min_length if min_length <= len(nums) else 0
```

#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        maxlength = 0
        used = {} # char: index of latest occurance of the char
        
        l = 0
        for r, char in enumerate(s):
            if char in used and l <= used[char]:
                l = used[char] +1
            else:
                maxlength = max(maxlength, r - l + 1)
    
            used[char] = r
            
        return maxlength
```

#### [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        for start, end in sorted(intervals, key = lambda i: i[0]):
            if res and start <= res[-1][1]:
                res[-1][1] = max(res[-1][1], end)
            else:
                res.append([start, end])
        return res
```

#### [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if nums is None or len(nums) == 0:
            return     

        insertPos = 0
        for num in nums:
            if num != 0:
                nums[insertPos] =1944. Number of Visible People in a Queue num    
                insertPos += 1

        while insertPos < len(nums):
            nums[insertPos] = 0
            insertPos += 1
```

#### [645. Set Mismatch](https://leetcode.com/problems/set-mismatch/)
```python
# Solution 1: O(N) time and O(N) space
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]
```
```python
# Solution 2: swap to right place (allow to modify the input)
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            while nums[i] != i + 1 and nums[nums[i] - 1] != nums[i]:
                right_pos = nums[i] - 1
                nums[i], nums[right_pos] = nums[right_pos], nums[i]
                
        for i in range(len(nums)):
            if nums[i] != i + 1:
                return [nums[i], i + 1]
```
```python
# Solution 3: negative marking (NOT allow to modify the input)
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        for num in nums:
            cur = abs(num)
            if nums[cur - 1] < 0:
                duplicate = cur
            else:
                nums[cur - 1] *= -1

        for i in range(len(nums)):
            if nums[i] > 0:
                missing = i + 1
            else:
                nums[i] = abs(nums[i]) # restore numbers

        return [duplicate, missing]

# Solution 4: sort and iterate
```

#### [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
```python
# Solution 1: swap
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while nums[i] > 0 and nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                right_pos = nums[i] - 1
                nums[i], nums[right_pos] = nums[right_pos], nums[i]
                
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
            
        return n + 1
```
```python
# Solution 2: negative marking
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        
        for i in range(n):
            if nums[i] <= 0: # remove useless elements
                nums[i] = n + 1
                
        for num in nums:
            curr = abs(num)
            if curr <= n and nums[curr - 1] > 0:
                nums[curr - 1] *= -1
            
        for i in range(n):
            if nums[i] >= 0:
                return i + 1
            
        return n + 1
```

#### [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
```python
# Solution 1: negative marking - O(n) time and O(1) space
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        for num in nums:
            cur = abs(num)
            if nums[cur] < 0:
                duplicate = cur
                break
            nums[cur] = - nums[cur]

        # Restore numbers
        for i in range(len(nums)):
            nums[i] = abs(nums[i])

        return duplicate
```
```python
# Solution 2: binary search - O(nlog) time and O (1) sapce
# e.g. count(1,2,3,4,5,6,7)= (1,2,3,6,7,8,8)
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        low = 1
        high = len(nums) - 1
        
        while low <= high:
            mid = low + (high - low) // 2

            # Count numbers less than or equal to 'mid'
            count = sum(num <= mid for num in nums)
            if count > mid:
                duplicate = mid
                high = mid - 1
            else:
                low = mid + 1
                
        return duplicate
```
We can rephrase the problem as finding the entrance point of a cyclic linkedlist, which is the same as [142. Linked List Cycle II](#142-Linked-List-Cycle-II)
In **Phrase 1**, fast pointer moves twice as fast as the slow pointer, until the two pointers meet. At intersection, we have
```
2 * (F + a) = F + nC + a, where n is some contant
```
Solving the eq. gives us
```
F + n = nC
```

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/287.%20Find%20the%20Duplicate%20Number.png" width="500">
   
In **Phrase 2**, let slow pointer start at the head and fast start at the intersection point, and they move at the same speed. They will meet at the entrance of cycle. To prove:
- The slow pointer started at zero, so its position after F steps is F.
- The faster pointer started at the intersection point F + a = nC, so its position after F steps is nC + F, that is the same point as F.
```python
# Solution 3: Floyd's Tortoise and Hare (Cycle Detection)
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # phrase 1
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # phrase 2
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        
        return fast    
```

#### [667. Beautiful Arrangement II](https://leetcode.com/problems/beautiful-arrangement-ii/)
use first k+1 elements to create list of k distinct diff, i.e. 1, 1+k, 2, k, 3, k-1, ...
e.g. [1,4,2,3,5,6] -> [1,4,2,3,5,6]
```python
class Solution:
    def constructArray(self, n: int, k: int) -> List[int]:
        res = [0] * n
        res[0] = 1
        
        i = 1
        intv = k
        while i <= k:
            if i % 2 == 1:
                res[i] = res[i - 1] + intv
            else:
                res[i] = res[i - 1] - intv
            i += 1
            intv -= 1
        
        for i in range(k + 1, n):
            res[i] = i + 1
            
        return res
```

#### [697. Degree of an Array](https://leetcode.com/problems/degree-of-an-array/)
```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        first = {} # index of first occurance of this number
        count = defaultdict(int)
        
        min_length = degree = 0
        for i, num in enumerate(nums):
            first.setdefault(num, i) # assign i to nums only if nums does not exist
            count[num] += 1
            
            if count[num] > degree:
                degree = count[num]
                min_length = i - first[num] + 1
            elif count[num] == degree:
                min_length = min(min_length, i - first[num] + 1)
                
        return min_length
```

#### [565. Array Nesting](https://leetcode.com/problems/array-nesting/)

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/565.%20Array%20Nesting.png" width="500">

```python
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        longest = 0
        for curr in nums:
            if curr == -1: 
                continue
                
            cnt = 0
            while nums[curr] != -1:
                cnt += 1
                nums[curr], curr = -1, nums[curr] # be careful with order
                
            longest = max(longest, cnt)
                
        return longest
```

#### [769. Max Chunks To Make Sorted](https://leetcode.com/problems/max-chunks-to-make-sorted/)
find some splitting line so that numbers on the left are smaller than numbers on the right. The idea is very similar to quick sort.
```python
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        chunks = 0
        left_max = arr[0]
        for idx in range(len(arr)):
            left_max = max(left_max, arr[idx])
            if left_max == idx:
                chunks += 1
        
        return chunks
```

#### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        
        L, R, answer = [0]*length, [0]*length, [0]*length
        
        L[0] = 1
        for i in range(1, length):
            L[i] = nums[i - 1] * L[i - 1]
        
        R[length - 1] = 1
        for i in reversed(range(length - 1)):
            R[i] = nums[i + 1] * R[i + 1]
        
        for i in range(length):
            answer[i] = L[i] * R[i]
        
        return answer
    
# space optimziation
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        answer = [0]*length
        answer[0] = 1
        
        for i in range(1, length):
            answer[i] = nums[i - 1] * answer[i - 1]
        
        R = 1
        for i in range(length-1, -1, -1):
            answer[i] = answer[i] * R
            R *= nums[i]
        
        return answer
```


## Two Pointers
* [167. Two Sum II](#167-Two-Sum-II)
* [15. 3Sum](#15-3Sum)
* [259. 3Sum Smaller](#259-3Sum-Smaller)
* [16. 3Sum Closest](#16-3Sum-Closest)
* [633. Sum of Square Numbers](#633-Sum-of-Square-Numbers)
* [345. Reverse Vowels of a String](#345-Reverse-Vowels-of-a-String)
* [680. Valid Palindrome II](#680-Valid-Palindrome-II)
* [88. Merge Sorted Array](#88-Merge-Sorted-Array)
* [141. Linked List Cycle](#141-Linked-List-Cycle)
* [524. Longest Word in Dictionary through Deleting](#524-Longest-Word-in-Dictionary-through-Deleting)

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

#### [15. 3Sum](https://leetcode.com/problems/3sum/?envType=study-plan-v2&envId=top-interview-150)
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []

        for i, n in enumerate(nums):
            if n > 0:
                break
            if i > 0 and n == nums[i - 1]:
                continue
            
            l, r = i + 1, len(nums) - 1
            while l < r:
                three_sum = n + nums[l] + nums[r]

                if three_sum < 0:
                    l += 1
                elif three_sum > 0: 
                    r -= 1
                else:
                    res.append([n, nums[l], nums[r]])
                    l += 1
                    r -= 1

                    while l < r and nums[l] == nums[l - 1]:
                        l += 1

        return res
```


#### [259. 3Sum Smaller](https://leetcode.com/problems/3sum-smaller/)
```python
# if (i,j,k) works, then (i,j,k), (i,j,k-1),..., (i,j,j+1) all work, (k-j) triplets
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        count = 0
        nums.sort()
        for i in range(len(nums)):
            j, k = i + 1, len(nums) - 1
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                if s < target:
                    count += k-j
                    j += 1
                else:
                    k -= 1
                    
        return count
```


#### [16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/description/)
```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        res = nums[0] + nums[1] + nums[2]
        nums.sort() # O(nlogn)
        
        for i in range(len(nums)-2): 
            #if i > 0 and nums[i]==nums[i-1]: continue 
                
            l, r = i+1, len(nums)-1 
            while l<r:
                total = nums[i] + nums[l] + nums[r]
                
                if total == target:
                    return total
                
                if abs(total - target) < abs(res - target):
                    res = total

                if total < target: 
                    l += 1
                elif total > target: 
                    r-=1
            
        return res
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
        if not head or not head.next:
            return False
        
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if fast and slow and slow == fast:
                return True
            
        return False
```
Time complexity : O(n), where n is the total number of nodes in the linked list. 
Consider the following two cases separately.
1. List has no cycle: The fast pointer reaches the end first and the run time depends on the list's length, which is O(n).
2. List has a cycle: Consider breaking down the movement of the slow pointer into two steps, the non-cyclic part (N nodes) when it does not enter the cycle yet and the cyclic part (K nodes):
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


## String
* [242. Valid Anagram](#242-Valid-Anagram)
* [409. Longest Palindrome](#409-Longest-Palindrome)
* [205. Isomorphic Strings](#205-Isomorphic-Strings)
* [647. Palindromic Substrings](#647-Palindromic-Substrings)
* [5. Longest Palindromic Substring](#5-Longest-Palindromic-Substring)
* [9. Palindrome Number](#9-Palindrome-Number)
* [696. Count Binary Substrings](#696-Count-Binary-Substrings)

Things to look out for during interviews:
- Ask about input character set and case sensitivity.
- note that Counter() takes O(1) space instead of O(n) because range of characters is usually bounded by 26, a constant.


#### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t)
```

#### [409. Longest Palindrome](https://leetcode.com/problems/longest-palindrome/description/)
```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        odds = sum(v & 1 for v in collections.Counter(s).values()) # v & 1 is equivalent to v%2==1
        return len(s) - odds + bool(odds)
```

#### [205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/description/)
```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mapping_s_t = {}
        mapping_t_s = {}
        
        for c1, c2 in zip(s, t):
            if (c1 not in mapping_s_t) and (c2 not in mapping_t_s):
                mapping_s_t[c1] = c2
                mapping_t_s[c2] = c1          
            elif mapping_s_t.get(c1) != c2 or mapping_t_s.get(c2) != c1:
                return False
            
        return True
```
For paper, the transformed string will be 01034. <br />
For title. The transformed string would be 01034, which is the same as that for paper
```python
class Solution: 
    def encode(self, s: str) -> str:
        index_mapping = {}
        encoded = []
        
        for i, c in enumerate(s):
            if c not in index_mapping:
                index_mapping[c] = str(i)
            encoded.append(index_mapping[c])
        
        return " ".join(encoded)
    
    def isIsomorphic(self, s: str, t: str) -> bool:
        return self.encode(s) == self.encode(t)
```
```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return len(set(zip(s, t))) == len(set(s)) == len(set(t))
```
Follow up question <br />
input: ['aab', 'xxy', 'xyz', 'abc', 'def', 'xyx'] 

return: <br />
[<br />
['xyx'], <br />
['xyz', 'abc', 'def'], <br />
['aab', 'xxy']<br />
]
```python
def groupIsomorphic(strs)
        def encode(s):
            d = {}
            encoded = []
            for c in s:
                if c not in d:
                    d[c] = len(d)
                encoded.append(d[c])
            return str(encoded)

        groups = {}
        for s in strs:
            encoded = encode(s)
            groups.get(encoded, []).append(s)

        return list(groups.values())
```

#### [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/description/)
a[i], a[i+1], ..., a[j-1], a[j] is a palindrome if a[i] == a[j] and <br />
1. a[i+1], ..., a[j-1] is a palindrome, or 
2. j-i < 3
   - i = j: we have only 1 character
   - i + 1 = j: we have 2 repeated character a[i] = a[j]
   - i + 2 = j: we have 2 repeated character a[i] = a[j] wrapping around one character (does not matter what it is)
We use dp[i+1][j-1] to calculate dp[i][j] because i is in descending order and j is in ascending order. <>br /
Thus we know the value of dp[i+1][j-1] before dp[i][j]
```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        res = 0
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                dp[i][j] = s[i] == s[j] and (j-i < 3 or dp[i+1][j-1])
                res += dp[i][j]
        return res
```
```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        for i in range(len(s)):
            count += self.EAC(s, i, i) # single character center
            count += self.EAC(s, i, i+1) # two characters center
        
        return count
            
    # Expand Around Center
    def EAC(self, s, l, r,):
        count = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
            
        return count
```

#### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
```python
class Solution:
    def longestPalindrome(self, s):
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        longest_palindrom = ''
        for i in range(n - 1, -1, -1):
            for j in range(i, n):  
                if s[i] == s[j] and (j - i < 3 or dp[i+1][j-1]):
                        dp[i][j] = 1
                        longest_palindrom = max(longest_palindrom, s[i:j+1], key = len)
                
        return longest_palindrom
```

#### [9. Palindrome Number](https://leetcode.com/problems/palindrome-number/)
```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False

        reverted = 0
        while x > reverted:
            reverted = reverted * 10 + x % 10
            x = x // 10
       
        return x == reverted or x == reverted //10 
        # if x is single center, e.g. 12321
```

#### [696. Count Binary Substrings](https://leetcode.com/problems/count-binary-substrings/)
```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        # group, e.g. if s = "110001111000000", then groups = [2, 3, 4, 6]
        groups = [1]
        for i in range(1, len(s)):
            if s[i-1] != s[i]:
                groups.append(1)
            else:
                groups[-1] += 1

        ans = 0
        for i in range(1, len(groups)):
            ans += min(groups[i-1], groups[i])
        return ans
```
```python
# space optimization
class Solution(object):
    def countBinarySubstrings(self, s):
        ans, prev, cur = 0, 0, 1
        for i in range(1, len(s)):
            if s[i-1] != s[i]:
                ans += min(prev, cur)
                prev, cur = cur, 1
            else:
                cur += 1

        return ans + min(prev, cur)
```


## Matrix
* [566. Reshape the Matrix](#566-Reshape-the-Matrix)
* [240. Search a 2D Matrix II](#240-Search-a-2D-Matrix-II)
* [378. Kth Smallest Element in a Sorted Matrix](#378-Kth-Smallest-Element-in-a-Sorted-Matrix)
* [373. Find K Pairs with Smallest Sums](#373-Find-K-Pairs-with-Smallest-Sums)
* [74. Search a 2D Matrix](#74-Search-a-2D-Matrix)
* [766. Toeplitz Matrix](#766-Toeplitz-Matrix)
    
    
#### [566. Reshape the Matrix](https://leetcode.com/problems/reshape-the-matrix/)
```python
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        
        if r * c != m * n: 
            return mat 
        
        count = 0
        ans = [[0] * c for _ in range(r)]
        for i in range(m):
            for j in range(n):
                row, col = divmod(count, c) 
                ans[row][col] = mat[i][j]
                count += 1
                
        return ans
# numpy has a function reshape: np.reshape(nums, (r, c)).tolist()
```

#### [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)

 <img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/240.%20Search%20a%202D%20Matrix%20II.png" width="300">

we seek along the matrix's middle column for an index row such that matrix[row-1][mid] < target < matrix[row][mid] <br /> 
The existing matrix can be partitioned into four submatrice around this index:
- the top-left and bottom-right submatrice cannot contain target, so we ignore
- the bottom-left and top-right submatrice are sorted two-dimensional matrices, so we can recursively apply this algorithm to them
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        def search(left, right, up, down):
            # base case: zero are
            if left > right or up > down:
                return False
            
            # base case: smaller than smallest or larger than largest matrix value
            if target < matrix[up][left] or target > matrix[down][right]:
                return False
            
            mid = left + (right - left) // 2
            
            row = up
            while row <= down and matrix[row][mid] <= target:
                if matrix[row][mid] == target:
                    return True
                row += 1
                
            return search(left, mid - 1, row, down) or search(mid + 1, right, up, row - 1)
        
        return search(0, len(matrix[0]) - 1, 0, len(matrix) - 1)
```
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # start our "pointer" in the bottom-left
        row = len(matrix) - 1
        col = 0

        while col < len(matrix[0]) and row >= 0:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                return True
        
        return False
```

#### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
A simpler version is to find Kth smallest element from 2 sorted lists using two pointers <br />
This problem can be reframed as finding the K smallest elements from amongst N sorted lists <br />
```
# # # # # ? .
# # # ? . . .
# ? . . . . .   "#" means pair already in the output
# ? . . . . .   "?" means pair currently in the queue
# ? . . . . .
? . . . . . .
. . . . . . .
```
```python
# Solution 1: min heap - let X = min(K, len(matrix)), it's O(KlogX) time and O(X) space
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        heap = []
        
        def push(i, j):
            if i < len(matrix) and j < len(matrix[0]):
                heapq.heappush(heap, [matrix[i][j], i, j])

        push(0, 0)
        while k:
            element, i, j = heapq.heappop(heap)
            push(i, j + 1)
            
            if j == 0:
                push(i + 1, 0)

            k -= 1

        return element
```
```python
# Solution 2: binary search
# let N = len(matrix), it's O(Nlog(max-min)) time and O(1) space
class Solution:
    def countLessEqual(self, matrix, mid, smaller, larger):
        n = len(matrix)
        row, col = n - 1, 0
        
        count = 0
        while row >= 0 and col < n:
            if matrix[row][col] >= mid:
                larger = min(larger, matrix[row][col])
                row -= 1
            else:
                smaller = max(smaller, matrix[row][col])
                count += row + 1
                col += 1

        return count, smaller, larger
    
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        start, end = matrix[0][0], matrix[n - 1][n - 1]
        while start < end:
            mid = start + (end - start) / 2
            
            smaller = matrix[0][0]  # track the biggest number less than or equal to the mid
            larger = matrix[n - 1][n - 1] # track the smallest number greater than the mid

            count, smaller, larger = self.countLessEqual(matrix, mid, smaller, larger)

            if count == k:
                return smaller
            if count < k:
                start = larger  # search higher
            else:
                end = smaller  # search lower

        return start
```

#### [373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)
This problem can be visualized as a m x n matrix. For example, for nums1=[1,7,11], and nums2=[2,4,6]
```
      2   4   6
   +------------
 1 |  3   5   7
 7 |  9  11  13
11 | 13  15  17
```
Then it becomes the same problem as [378. Kth Smallest Element in a Sorted Matrix](#378-Kth-Smallest-Element-in-a-Sorted-Matrix)
```python
# let X = min(K, len(nums1)), it's O(KlogX) time: K iterations of popping and pushing from a heap of X elements
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = []
        
        def push(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(heap, [nums1[i] + nums2[j], i, j])
                
        push(0, 0)
        res = []
        while heap and k:
            _, i, j = heapq.heappop(heap)
            res.append([nums1[i], nums2[j]])
            push(i, j + 1)
            
            if j == 0:
                push(i + 1, 0)
                
            k -= 1
                
        return res
```

#### [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
```python
# O(log(mn)) time and O(1) space
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix[0])
        start, end = 0, n * len(matrix) - 1
        
        while start <= end:
            mid_idx = start + (end - start) // 2
            mid_element = matrix[mid_idx // n][mid_idx % n]
            
            if target == mid_element:
                return True
            elif target < mid_element:
                end = mid_idx - 1
            else:
                start = mid_idx + 1
                
        return False
```

#### [766. Toeplitz Matrix](https://leetcode.com/problems/toeplitz-matrix/)
```python
# brute force: O(MN) time and O(M + N) space
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        # on the same diagonal, r1 - c1 == r2 - c2.
        groups = {}
        for r, row in enumerate(matrix):
            for c, val in enumerate(row):
                if r - c not in groups:
                    groups[r - c] = val
                elif groups[r - c] != val:
                    return False
        return True
    
# space optimization
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        return all(r == 0 or c == 0 or 
                   matrix[r - 1][c - 1] == val
                   for r, row in enumerate(matrix)
                   for c, val in enumerate(row))
```


## LinkedList
* [160. Intersection of Two Linked Lists](#160-Intersection-of-Two-Linked-Lists)
* [206. Reverse Linked List](#206-Reverse-Linked-List)
* [21. Merge Two Sorted Lists](#21-Merge-Two-Sorted-Lists)
* [83. Remove Duplicates from Sorted List](#83-Remove-Duplsrcicates-from-Sorted-List)
* [19. Remove Nth Node From End of List](#19-Remove-Nth-Node-From-End-of-List)
* [24. Swap Nodes in Pairs](#24-Swap-Nodes-in-Pairs)
* [445. Add Two Numbers II](#445-Add-Two-Numbers-II)
* [234. Palindrome Linked List](#234-Palindrome-Linked-List)
* [725. Split Linked List in Parts](#725-Split-Linked-List-in-Parts)
* [328. Odd Even Linked List](#328-Odd-Even-Linked-List)
* [142. Linked List Cycle II](#142-Linked-List-Cycle-II)


#### [160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/description/)
Imagine that we have two linked lists, A and B, and we know that their lengths are N and M respectively, where M = 8 > N = 5. <br />
Because the "tails" must have the same length, if there is an intersection, then the intersection node will be one of these 5 possibilities. <br />
Thus, we would start by setting a pointer at the start of the shorter list, and a pointer at the first possible matching node of the longer list, which is |M| - |N|
 = 3. <br />
Then, we just need to step the two pointers through the list, each time checking whether or not the nodes are the same. <br />
 
 <img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/160.%20Intersection%20of%20Two%20Linked%20Lists.png" width="700">

We can implement this algorithm through one pass: 

Suppose that c is the shared part, and a, b are exclusive parts of list A and B, i.e. A = a + c, B = b + c. <br />
Let's set two points to step through A + B = (a + c) + (b + c) and B + A = (b + c) + (a + c), <br />
Since a + c + b = b + c + a, the two pointers will meet at the start point of c, if there's intersection

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pA = headA
        pB = headB

        while pA != pB:
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next

        return pA
```
Note: In the case lists do not intersect, the pointers for A and B will still line up and reach their respective ends at the same time

#### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
python inline swap: a, b = b, a
1. the right-hand side of '=', i.e. a, b, are created in memory. Note, no assignment is made yet
2. the left-hand side of '=' are assigned values in the order of left-to-right, i.e a first and b next
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# iterative - O(n) time and O(1) space
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
            
        return prev
```
```python
# recursive 1 - O(n) time and O(n) space
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if (not head) or (not head.next):
            return head
        
        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None

        return p # p is always the last node in original list, or new head in new list
```
```python
# recursive 2 - this is actually the same as the iterative solution
class Solution:
    def reverseList(self, curr: Optional[ListNode], prev = None) -> Optional[ListNode]:
        if not curr:
            return prev
        
        new_head = curr.next
        curr.next = prev
        
        return self.reverseList(new_head, curr)
```

### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
```python
# iterative - O(n) time and O(1) space
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        curr = dummy = ListNode()

        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next            
            curr = curr.next

        # append the rest of l1 or l2
        curr.next = l1 or l2

        return dummy.next
```
```python
# recursive (dp) - O(n) time and O(n) space
class Solution:
    def mergeTwoLists(self, l1, l2):
        if l1 is None:
            return l2
        
        if l2 is None:
            return l1
        
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2        
```

#### [83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/)
```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        
        while curr and curr.next:
            if curr.next.val == curr.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
                
        return head
```

#### [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        fast = slow = dummy = ListNode()
        
        dummy.next = head
        for _ in range(n + 1):
            fast = fast.next
            
        while fast:
            fast = fast.next
            slow = slow.next
            
        slow.next = slow.next.next
        
        return dummy.next
```

#### [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
```python
# recursive
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:

        # If the list has no node or has only one node left.
        if not head or not head.next:
            return head

        first = head
        second = head.next

        first.next  = self.swapPairs(second.next)
        second.next = first

        return second
```
```python
# iterative
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = prev = ListNode()
        prev.next = head
        
        while prev.next and prev.next.next:
            first = prev.next
            second = first.next
            
            prev.next, second.next, first.next = second, first, second.next
            prev = first
            
        return dummy.next
```

#### [445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/description/)
```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head
        
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
            
        return prev
    
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        l1 = self.reverseList(l1)
        l2 = self.reverseList(l2)
        
        prev = None
        carry = 0
        while l1 or l2:
            x1 = l1.val if l1 else 0
            x2 = l2.val if l2 else 0
            
            carry, val = divmod(carry + x1 + x2, 10)
            
            curr = ListNode(val)
            curr.next = prev
            prev = curr
            
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        if carry:
            curr = ListNode(carry)
            curr.next = prev
            prev = curr

        return prev
```

#### [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if head is None:
            return True

        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        p1 = head
        p2 = second_half_start
        while p2: # l2 is equal to or smaller than l1
            if p1.val != p2.val:
                return False
            p1 = p1.next
            p2 = p2.next

        return True    

    def end_of_first_half(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow # if length of list is odd, slow will stop at the mid point
    
    def reverse_list(self, head):
        prev, curr = None, head
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        return prev
```

#### [725. Split Linked List in Parts](https://leetcode.com/problems/split-linked-list-in-parts/)
```python
class Solution:
    def splitListToParts(self, root: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        curr, length = root, 0
        while curr:
            curr, length = curr.next, length + 1
            
        chunk_size, rest = length // k, length % k # distribute the rest to each chunk
        res = [chunk_size + 1] * rest + [chunk_size] * (k - rest)
        
        prev, curr = None, root
        for index, num in enumerate(res):

            res[index] = curr
            for i in range(num - 1):
                curr = curr.next
                
            if curr:
                curr.next, curr = None, curr.next
                
        return res
```

#### [328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)
```python
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        
        odd = head 
        even = head.next
        even_head = even
        
        while even and even.next: 
            odd.next = odd.next.next
            even.next = even.next.next
           
            odd = odd.next
            even = even.next
        
        odd.next = even_head
        
        return head
```

#### [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
In **Phrase 1**, fast pointer moves twice as fast as the slow pointer, until the two pointers meet. At intersection, we have
```
2 * (F + a) = F + nC + a, where n is some contant
```
Solving the eq. gives us
```
F + n = nC
```

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/142.%20Linked%20List%20Cycle%20II.png" width="500">
   
In **Phrase 2**, let slow pointer start at the head and fast start at the intersection point, and they move at the same speed. They will meet at the entrance of cycle. To prove:
- The slow pointer started at zero, so its position after F steps is F.
- The faster pointer started at the intersection point F + a = nC, so its position after F steps is nC + F, that is the same point as F.
```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # phrase 1
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        else: # fast or fast.next is None
            return None
        
        # phrase 2
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return fast
```
If the condition of the "while" loop becomes False, it will enter the "else" statement <br />
But if the "break" condition is met first, "else" statment won't be executed

## Tree
* [BFS](#BFS)
	* [637. Average of Levels in Binary Tree](#637-Average-of-Levels-in-Binary-Tree)
	* [513. Find Bottom Left Tree Value](#513-Find-Bottom-Left-Tree-Value)
* [DFS](#DFS)
	* [144. Binary Tree Preorder Traversal](#144-Binary-Tree-Preorder-Traversal)
	* [145. Binary Tree Postorder Traversal](#145-Binary-Tree-Postorder-Traversal)
	* [94. Binary Tree Inorder Traversal](#94-Binary-Tree-Inorder-Traversal)
* [Recursion](#Recursion)
	* [104. Maximum Depth of Binary Tree](#104-Maximum-Depth-of-Binary-Tree)
	* [111. Minimum Depth of Binary Tree](#111-Minimum-Depth-of-Binary-Tree)
	* [110. Balanced Binary Tree](#110-Balanced-Binary-Tree)
	* [543. Diameter of Binary Tree](#543-Diameter-of-Binary-Tree)
	* [687. Longest Univalue Path](#687-Longest-Univalue-Path)
	* [226. Invert Binary Tree](#226-Invert-Binary-Tree)
	* [617. Merge Two Binary Trees](#617-Merge-Two-Binary-Trees)
	* [112. Path Sum](#112-Path-Sum)
	* [437. Path Sum III](#437-Path-Sum-III)
	* [572. Subtree of Another Tree](#572-Subtree-of-Another-Tree)
	* [101. Symmetric Tree](#101-Symmetric-Tree)
	* [404. Sum of Left Leaves](#404-Sum-of-Left-Leaves)
	* [337. House Robber III](#337-House-Robber-III)
	* [671. Second Minimum Node In a Binary Tree](#671-Second-Minimum-Node-In-a-Binary-Tree)
* [BST](#BST)
	* [669. Trim a Binary Search Tree](#669-Trim-a-Binary-Search-Tree)
	* [230. Kth Smallest Element in a BST](#230-Kth-Smallest-Element-in-a-BST)
	* [701. Insert into a Binary Search Tree](#701-Insert-into-a-Binary-Search-Tree)
	* [450. Delete Node in a BST](#450-Delete-Node-in-a-BST)
	* [1382. Balance a Binary Search Tree](#1382-Balance-a-Binary-Search-Tree)
	* [538. Convert BST to Greater Tree](#538-Convert-BST-to-Greater-Tree)
	* [235. Lowest Common Ancestor of a Binary Search Tree](#235-Lowest-Common-Ancestor-of-a-Binary-Search-Tree)
	* [236. Lowest Common Ancestor of a Binary Tree](#236-Lowest-Common-Ancestor-of-a-Binary-Tree)
	* [108. Convert Sorted Array to Binary Search Tree](#108-Convert-Sorted-Array-to-Binary-Search-Tree)
	* [109. Convert Sorted List to Binary Search Tree](#109-Convert-Sorted-List-to-Binary-Search-Tree)
	* [653. Two Sum IV Input is a BST](#653-Two-Sum-IV-Input-is-a-BST)
	* [530. Minimum Absolute Difference in BST](#530-Minimum-Absolute-Difference-in-BST)
	* [501. Find Mode in Binary Search Tree](#501-Find-Mode-in-Binary-Search-Tree)
* [Trie](#Trie)
	* [208. Implement Trie](#208-Implement-Trie)
	* [677. Map Sum Pairs](#677-Map-Sum-Pairs)

### BFS
#### [637. Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)
```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# BFS
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        if not root:
            return 0
        
        res = []
        q = deque([root])
        while q:
            size = len(q)
            
            level_sum = 0
            for _ in range(size):
                node = q.popleft()
                
                level_sum += node.val

                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                
            res.append(level_sum/size)
            
        return res
```
```python
# DFS
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:

        def traverse(node, depth = 0):
            if node:
                if len(res) <= depth:
                    res.append([0, 0])

                res[depth][0] += node.val
                res[depth][1] += 1

                traverse(node.left, depth + 1)
                traverse(node.right, depth + 1)
                   
        res = [] # [sum, number of nodes]
        traverse(root)
        return [s/n for s, n in res]
```

#### [513. Find Bottom Left Tree Value](https://leetcode.com/problems/find-bottom-left-tree-value/)
```python
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        
        q = deque([root])
        while q:
            node = q.popleft()
            if node.right:
                q.append(node.right)
            if node.left:
                q.append(node.left)
            
        return node.val
```

### DFS

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/Traversal.png" width="500">

#### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
pre-order: root -> left -> right
```python
# recursive DFS
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right) if root else []
```
```python
# iterative DFS
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        stack = [root]
        output = []
        while stack:
            node = stack.pop()
            if node:
                output.append(node.val)
                stack.append(node.right)  # append right first, left next
                stack.append(node.left)   # so left comes out first when pop
        
        return output
```
```python
# Morris Taversal
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        output = []
        curr = root
        while curr:  
            if not curr.left: 
                output.append(curr.val)
                curr = curr.right 
            else: 
                predecessor = curr.left 

                # -- FIND PREDECESSOR --
                while predecessor.right and predecessor.right is not curr: 
                    predecessor = predecessor.right 

                # -- CREATE VIRTUAL LINKS --
                if not predecessor.right:
                    output.append(curr.val)
                    predecessor.right = curr  
                    curr = curr.left  
                    
                # -- RESTORE TREE --
                else:
                    predecessor.right = None
                    curr = curr.right         

        return output
```

#### [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)
post-order: left -> right-> root
```python
# recursive
class Solution:
    def postorderTraversal(self, root):    
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val] if root else []
```
```python
# modified preorder: post order is the reverse of right-first preorder (root -> right -> left)
class Solution:
    def postorderTraversal(self, root):
        output = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                output.append(node.val)
                stack.append(node.left)
                stack.append(node.right)

        return output[::-1]
```
```python
# flag of visit
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        output = []
        stack = [(root, False)]
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    output.append(node.val)
                else:
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))

        return output
```
```python
# Morris Traversal
class Solution:
    def postorderTraversal(self, root):
        output = []
        curr = root
        while curr:  
            if not curr.right: 
                output.append(curr.val)
                curr = curr.left 
            else: 
                predecessor = curr.right 

                # -- FIND PREDECESSOR --
                while predecessor.left and predecessor.left is not curr: 
                    predecessor = predecessor.left 

                # -- CREATE VIRTUAL LINKS --
                if not predecessor.left:
                    output.append(curr.val)
                    predecessor.left = curr  
                    curr = curr.right  
                    
                # -- RESTORE TREE --
                else:
                    predecessor.left = None
                    curr = curr.left         

        return output[::-1]
```

#### [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
[graph illustration](https://leetcode.com/problems/binary-tree-inorder-traversal/solution/)
```python
# recursive
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
         return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right) if root else []
```
```python        
# iterative
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        output = []
        
        curr = root
        while stack or curr:
            while curr:
                stack.append(curr)
                curr = curr.left
                
            curr = stack.pop()
            output.append(curr.val)
            curr = curr.right
        
        return output 
```
```python
# flag of visit
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        output = []
        stack = [(root, False)]
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    output.append(node.val)
                else:
                    stack.append((node.right, False))
                    stack.append((node, True))
                    stack.append((node.left, False))

        return output
```
```python
# Morris Traversal
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        output = []
        curr = root
        while curr:
            if not curr.left:
                output.append(curr.val)
                curr = curr.right
            else:
                predecessor = curr.left
                
                # -- FIND PREDECESSOR --
                while predecessor.right and predecessor.right is not curr:
                    predecessor = predecessor.right
                    
                # -- CREATE VIRTUAL LINKS --
                if not predecessor.right:
                    predecessor.right = curr
                    curr = curr.left
                    
                # -- RESTORE TREE --
                else:
                    output.append(curr.val)
                    predecessor.right = None
                    curr = curr.right
                
        return output
```

### Recursion
#### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)
```python
# recursive DFS - O(n) time and O(logN) ~ O(N) space
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```
```python 
# iterative BFS - O(n) time and O(logN) ~ O(N) space
from collections import deque
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        q = deque([root])
        levels = 0
        while q:
            levels += 1
            size = len(q)
            
            for _ in range(size):
                node = q.popleft()

                if node.left:
                    q.append(node.left)

                if node.right:
                    q.append(node.right)
                
        return levels
```
```python
# iterative DFS
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        stack = []
        if root:
            stack.append((1, root))
        
        depth = 0
        while stack:
            current_depth, root = stack.pop()
            if root:
                depth = max(depth, current_depth)
                stack.append((current_depth + 1, root.left))
                stack.append((current_depth + 1, root.right))
        
        return depth
```

#### [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
```python
# recursive
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        if not root.left:
            return 1 + self.minDepth(root.right)
        if not root.right:
            return 1 + self.minDepth(root.left)
        
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```
```python
# iterative BFS
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        q = deque([root])
        levels = 0
        while q:
            levels += 1
            size = len(q)
            
            for _ in range(size):
                node = q.popleft()
                
                if node:
                    if not node.left and not node.right:
                        return levels
                    else:
                        q.append(node.left)
                        q.append(node.right)
```
```python
# iterative DFS
class Solution:
    def minDepth(self, root):
        if not root:
            return 0
        
        stack = [(1, root)]
        min_depth = float('inf')
        
        while stack:
            depth, node = stack.pop()
            
            if node:
                if not node.left and not node.right:
                    min_depth = min(depth, min_depth)
                else:
                    stack.append((depth + 1, node.left))
                    stack.append((depth + 1, node.right))
        
        return min_depth 
```

#### [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
```python
# top down (pre-order) - O(nlogn) time and O(n) space
# each node at depth d is called d times, where d = height of the tree = logn
class Solution:
    def height(self, root: TreeNode) -> int:
        if not root:
            return -1
        
        return 1 + max(self.height(root.left), self.height(root.right))
    
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        left = self.height(root.left)
        right = self.height(root.right)

        return abs(left - right) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
```
```python
# bottom up (post-order) - O(n) time and O(n) space
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.isBalancedHelper(root)[0]
            
    def isBalancedHelper(self, root: TreeNode) -> (bool, int):
        # An empty tree is balanced and has height -1
        if not root:
            return True, -1
        
        leftIsBalanced, leftHeight = self.isBalancedHelper(root.left)
        if not leftIsBalanced:
            return False, 0
        
        rightIsBalanced, rightHeight = self.isBalancedHelper(root.right)
        if not rightIsBalanced:
            return False, 0
        
        return (abs(leftHeight - rightHeight) <= 1), 1 + max(leftHeight, rightHeight)
```

#### [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/description/)
 
 <img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/543.%20Diameter%20of%20Binary%20Tree.png" width="500">

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:

        def longest_path(node):
            if not node:
                return 0
            
            nonlocal diameter
            
            left_path = longest_path(node.left)
            right_path = longest_path(node.right)

            diameter = max(diameter, left_path + right_path)

            return max(left_path, right_path) + 1

        diameter = 0
        longest_path(root)
        
        return diameter
```

#### [687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/)
```python
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:

        def traversal(node):
            if not node: 
                return 0
            
            left_length = traversal(node.left)
            right_length = traversal(node.right)
            
            # if curr value != child value, previous length should not be added, and we reset left/right to 0 
            left = left_length + 1 if node.left and node.left.val == node.val else 0
            right = right_length + 1 if node.right and node.right.val == node.val else 0
            
            self.max_length = max(self.max_length, left + right)
            return max(left, right)

        self.max_length = 0
        traversal(root)
        
        return self.max_length
```

#### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/)
```python
# recursive - O(n) time and O(logn) ~ O(n) space
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)

        return root
```
```python
# iterative BFS - O(n) time and O(logn) ~ O(n) space
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        
        q = deque([root])
        while q:
            node = q.popleft() # changing it to pop() wil turn it to DFS, which works too
            if node:
                node.left, node.right = node.right, node.left
                q.append(node.left)
                q.append(node.right)
        
        return root      
```

#### [617. Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/submissions/)
```python
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if root1 and root2:
            root = TreeNode(root1.val + root2.val)
            root.left = self.mergeTrees(root1.left, root2.left)
            root.right = self.mergeTrees(root1.right, root2.right)
            
            return root
        else:
            return root1 or root2
```

#### [112. Path Sum](https://leetcode.com/problems/path-sum/description/)
```python
# recursive
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root: # this is initial check of input root, not base case
            return False

        targetSum -= root.val
        
        if not root.left and not root.right:  # reach a leaf
            return targetSum == 0
        
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
```
```python
# iterative
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False

        stack = [(root, targetSum - root.val)]
        while stack:
            node, targetSum = stack.pop()
            
            if not node.left and not node.right and targetSum == 0:  
                return True
            
            if node.right:
                stack.append((node.right, targetSum - node.right.val))
            if node.left:
                stack.append((node.left, targetSum - node.left.val))
                
        return False
```

#### [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/description/)
use **prefix sum** to find number of continuous subarrays that sum to Target in one pass O(n): [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/submissions/)
 ```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        
        def preorder(node: TreeNode, curr_sum) -> None:
            nonlocal count
            
            if not node:
                return 
            
            curr_sum += node.val
            count += dic[curr_sum - targetSum]
            dic[curr_sum] += 1
            
            preorder(node.left, curr_sum)
            preorder(node.right, curr_sum)
            
            # remove the current sum from the hashmap when going one node back
            dic[curr_sum] -= 1
            
        count = 0
        dic = defaultdict(int)
        dic[0] = 1
        
        preorder(root, 0)
        return count
 ```
 
 #### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)
```python
# O(m*n) time and O(logm) ~ O(m) space 
class Solution:
    def isSubtree(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
        if s is None:
            return False
        
        return self.isSubtreeWithRoot(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t);

    def isSubtreeWithRoot(self, s, t):
        if t is None and s is None:
            return True
        if t is None or s is None: 
            return False
        if t.val != s.val:
            return False
        
        return self.isSubtreeWithRoot(s.left, t.left) and self.isSubtreeWithRoot(s.right, t.right)
```

#### [101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
```python
# recursive
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self.checkSymmetric(root.left, root.right)
    
    def checkSymmetric(self, t1, t2):
        if t1 is None and t2 is None:
            return True
        if t1 is None or t2 is None:
            return False
        return t1.val == t2.val and self.checkSymmetric(t1.left, t2.right) and self.checkSymmetric(t1.right, t2.left)
```
```python
# iterative
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        q = deque([root, root])
        
        while q:
            t1 = q.popleft()
            t2 = q.popleft()
            
            if t1 is None and t2 is None:
                continue
            if t1 is None or t2 is None or t1.val != t2.val:
                return False
            
            q.append(t1.left)
            q.append(t2.right)
            q.append(t1.right)
            q.append(t2.left)
        
        return True
```

#### [404. Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves/description/)
```python
# iterative DFS (pre-order)
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if root is None: 
            return 0

        stack = [(root, False)]
        total = 0
        while stack:
            node, is_left = stack.pop() # changing it to popleft() turns it to BFS
            
            if is_left and node.left is None and node.right is None:
                total += node.val
                
            if node.right:
                stack.append((node.right, False))
                
            if node.left:
                stack.append((node.left, True))

        return total
```
```python
# recursive DFS (pre-order)
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if root is None:
            return 0

        def process_subtree(node, is_left):
            if node.left is None and node.right is None:
                return node.val if is_left else 0
            
            total = 0
            if node.left:
                total += process_subtree(node.left, True)
                
            if node.right:
                total += process_subtree(node.right, False)
                
            return total
        
        return process_subtree(root, False)
```
Morris Traversal explanation tutorial: [Youtube](https://www.youtube.com/watch?v=BuI-EULsz0Y)

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/Morris%20Traversal.png" width="350">

```python
# Morris Traversal (pre-order)
class Solution:
    def sumOfLeftLeaves(self, root):
        total_sum = 0
        curr = root
        while curr:
            # If there is no left child, we can simply explore the right subtree
            # without worrying about keeping track of currentNode's other children.
            if not curr.left: 
                curr = curr.right 
            else: 
                predecessor = curr.left 
                
                if predecessor.left is None and predecessor.right is None:
                    total_sum += predecessor.val
                    
                # Find the inorder predecessor for currentNode (predecessor is the rightmost tree of left subtree)
                while predecessor.right and predecessor.right is not curr:
                    predecessor = predecessor.right
                
                # -- CREATE VIRTUAL LINKS --
                # We've not yet visited the inorder predecessor, so we still need to explore currentNode's left subtree.
                # Before that, we will put a link back so that we can get back to the right subtree later
                if not predecessor.right:
                    predecessor.right = curr  
                    curr = curr.left 
                    
                # -- RESTORE TREE --
                # We have already visited the inorder predecessor, so we remove the virtual link we added
                # and then move onto the right subtree and explore it.
                else:
                    predecessor.right = None
                    curr = curr.right
                    
        return total_sum
```

#### [337. House Robber III](https://leetcode.com/problems/house-robber-iii/)
```python
# bottom up
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def helper(node):
            if not node:
                return (0, 0)
            
            left = helper(node.left)
            right = helper(node.right)
            
            
            rob = node.val + left[1] + right[1] # if we rob this node, we cannot rob its children
            not_rob = max(left) + max(right) # else we could choose to either rob its children or not
            
            return [rob, not_rob]

        return max(helper(root))
```
```python
# top down with memo
class Solution:
    def rob(self, root: TreeNode) -> int:
        rob_saved = {}
        not_rob_saved = {}

        def helper(node, parent_robbed):
            if not node:
                return 0

            if parent_robbed:
                if node in rob_saved:
                    return rob_saved[node]
                
                best = helper(node.left, False) + helper(node.right, False)
                rob_saved[node] = best
                return best
            else:
                if node in not_rob_saved:
                    return not_rob_saved[node]
                
                rob = node.val + helper(node.left, True) + helper(node.right, True)
                not_rob = helper(node.left, False) + helper(node.right, False)
                
                best = max(rob, not_rob)
                not_rob_saved[node] = best
                return best

        return helper(root, False)
```

#### [671. Second Minimum Node In a Binary Tree](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/description/)
```python
class Solution:
    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:

        def traverse(node):
            if node:
                if root.val < node.val < self.ans:
                    self.ans = node.val
                elif node.val == root.val:
                    traverse(node.left)
                    traverse(node.right)

        self.ans = float('inf')
        traverse(root)
        
        return self.ans if self.ans < float('inf') else -1
```

### BST
left <= root <= right

#### [669. Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree/)
```python
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        def trim(node):
            if not node:
                return None
            elif node.val > high:
                return trim(node.left)
            elif node.val < low:
                return trim(node.right)
            else:
                node.left = trim(node.left)
                node.right = trim(node.right)
                return node

        return trim(root)
```

#### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
```python
# recursive inorder - O(N) time and O(N) space
class Solution:
    def kthSmallest(self, root, k):
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
    
        return inorder(root)[k - 1]
```
```python
# recursive inorder with early stop
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(node):
            if node:
                inorder(node.left)
                self.k -= 1
                if self.k == 0:
                    self.val = node.val
                    return

                inorder(node.right)
        
        self.k, self.val = k, 0
        inorder(root)
        return self.val
```
```python        
# iterative inorder - O(H) time and O(H) space, where H is the tree height
class Solution:
    def kthSmallest(self, root, k):
        stack = []
        curr = root
        while True:
            while curr:
                stack.append(curr)
                curr = curr.left
                
            curr = stack.pop()
            
            k -= 1
            if k == 0:
                return curr.val
            
            curr = curr.right
```    

#### [701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
```python
# recursive
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        
        if val > root.val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)
            
        return root
```
```python
# iterative
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        node = root
        while node:
            # insert into the right subtree
            if val > node.val:
                if not node.right:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            # insert into the left subtree
            else:
                if not node.left:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
        return TreeNode(val)
```

#### [450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)
```python
class Solution:
    def find_min(self, root):
        root = root.right
        while root.left:
            root = root.left
        return root
    
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None
        
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                smallest = self.find_min(root)
                smallest.left = root.left
                
                return root.right
            
        return root
```

#### [1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)
```python
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        def inorder(node):
            if node:
                inorder(node.left)
                sort_array.append(node.val)
                inorder(node.right)
        
        def build_tree(sort_array):
            if not sort_array:
                return None
            
            mid = len(sort_array) // 2
            root = TreeNode(sort_array[mid])
            root.left = build_tree(sort_array[:mid]) # using pointers instead of passing the array will be more efficient
            root.right = build_tree(sort_array[mid + 1:])
            
            return root
        
        sort_array = []
        inorder(root)
        
        return build_tree(sort_array)
```

#### [538. Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)
```python
# recursive
class Solution:
    def __init__(self):
        self.total = 0

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            self.convertBST(root.right)
            self.total += root.val
            root.val = self.total
            self.convertBST(root.left)
            
        return root
```
```python
# iterative
class Solution:
     def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        stack = []
        total = 0
        
        curr = root
        while stack or curr:
            while curr:
                stack.append(curr)
                curr = curr.right
                
            curr = stack.pop()
            total += curr.val
            curr.val = total
            curr = curr.left
        
        return root 
```

#### [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
```python
# recursive
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > root.val and q.val > root.val:    
            return self.lowestCommonAncestor(root.right, p, q)    
        elif p.val < root.val and q.val < root.val:    
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```
```python
# iterative
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        while root:
            if p.val > root.val and q.val > root.val:    
                root = root.right  
            elif p.val < root.val and q.val < root.val:    
                root = root.left
            else:
                return root
```

#### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
```python
# recursive - O(n) time and O(n) space
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return None
        
        left_res = self.lowestCommonAncestor(root.left, p, q)
        right_res = self.lowestCommonAncestor(root.right, p, q)
        
        if (left_res and right_res) or (root in [p, q]):
            return root
        else:
            return left_res or right_res
```
```python
# recursive - O(n) time and O(n) space
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        stack = [root]
        parent = {root: None}
        # look for p and q
        while p not in parent or q not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
                
        ancestors = set()
        # put p and ancestors of p to the ancestors set
        while p:
            ancestors.add(p)
            p = parent[p]
        
        # walk through ancestors of q until that ancestor is also ancestor of p
        while q not in ancestors:
            q = parent[q]
            
        return q
```

#### [108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        
        def preorder(left, right):
            if left > right:
                return None

            p = left + (right - left) // 2

            root = TreeNode(nums[p])
            root.left = preorder(left, p - 1)
            root.right = preorder(p + 1, right)
            return root
        
        return preorder(0, len(nums) - 1)
```

#### [109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)
```python
# recursive and two pointers - O(nlogn) time and O(logn) space
class Solution:
    def findMiddle(self, head):
        # 12345 -> 12/3/45
        # 123456 -> 123/4/56
        prev = None
        slow = fast = head
        
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        if prev:
            prev.next = None

        return slow


    def sortedListToBST(self, head):
        if not head:
            return None
        
        # Base case: there is just one element in the linked list
        if head.next is None:
            return TreeNode(head.val)

        mid = self.findMiddle(head)
        node = TreeNode(mid.val)

        node.left = self.sortedListToBST(head)
        node.right = self.sortedListToBST(mid.next)
        
        return node
```
an alternative solution will be converting the linkedlist to an array first, and then solve it the same way as Question [108. Convert Sorted Array to Binary Search Tree](#108-Convert-Sorted-Array-to-Binary-Search-Tree), which has O(n) time and O(n) space. These two solutions form a typical time-space tradeoff.

Inorder traversal simulation: [Graph illustration](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/solution/)
```python
# inorder traversal
class Solution:    
    def getSize(self, head):
        ptr = head
        c = 0
        while ptr:
            ptr = ptr.next
            c += 1
        return c

    def sortedListToBST(self, head: ListNode) -> TreeNode:
        size = self.getSize(head)

        def inorder(l, r):
            nonlocal head

            if l > r:
                return None

            mid = l + (r - l) // 2

            left = inorder(l, mid - 1)

            node = TreeNode(head.val)   
            node.left = left

            head = head.next

            node.right = inorder(mid + 1, r)
            
            return node
        
        return inorder(0, size - 1)
```
note that we don't really find out the middle node of the linked list. We just have a variable telling us the index of the middle element. We simply need this to make recursive calls on the two halves.

#### [653. Two Sum IV Input is a BST](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/description/)
```python
# set - O(n) time and O(n) space
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        node_set = set()
        queue = [root]
        while queue:
            node = queue.pop()
            if node:
                if node.val in node_set:
                    return True

                node_set.add(k - node.val)
                queue.append(node.right)
                queue.append(node.left)

        return False     
```
```python
# inorder sorted array and two pointers - O(n) time and O(n) space
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        def inorder(root):
            return inorder(root.left) + [root.val] + inorder(root.right) if root else []
       
        numbers = inorder(root)
        
        l, r = 0, len(numbers) - 1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == k:
                return True
            elif s > k:
                r -= 1
            else:
                l += 1
```
```python
# inorder traversal space optimziation - O(n) time and O(logn) space
class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        def pushLeft(st, root):
            while root:
                st.append(root)
                root = root.left

        def pushRight(st, root):
            while root:
                st.append(root)
                root = root.right

        def nextLeft(st):
            node = st.pop()
            pushLeft(st, node.right)
            return node.val

        def nextRight(st):
            node = st.pop()
            pushRight(st, node.left)
            return node.val

        stLeft, stRight = [], [] # stack
        pushLeft(stLeft, root)
        pushRight(stRight, root)

        left, right = nextLeft(stLeft), nextRight(stRight)
        while left < right:
            if left + right == k: 
                return True
            if left + right < k:
                left = nextLeft(stLeft)
            else:
                right = nextRight(stRight)
                
        return False
```

#### [530. Minimum Absolute Difference in BST](https://leetcode.com/problems/minimum-absolute-difference-in-bst/)
```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def inorder(node):
            if not node:
                return
            
            inorder(node.left)
            
            if self.prev:
                self.min_diff = min(self.min_diff, node.val - self.prev.val)
            self.prev = node

            inorder(node.right)

        self.min_diff = float('inf')
        self.prev = None
        inorder(root)
        
        return self.min_diff
```

#### [501. Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree/description/)
```python
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        def inOrder(node):
            if node is None:
                return
            
            inOrder(node.left)
            if self.preNode:
                if self.preNode.val == node.val: 
                    self.curCnt += 1
                else:
                    self.curCnt = 1
        
            if self.curCnt > self.maxCnt:
                self.maxCnt = self.curCnt
                self.maxCntNums.clear()
                self.maxCntNums.append(node.val)
            elif self.curCnt == self.maxCnt:
                self.maxCntNums.append(node.val)
                
            self.preNode = node
            inOrder(node.right)
            
            
        self.maxCntNums = []
        self.curCnt = 1
        self.maxCnt = 1
        self.preNode = None
        inOrder(root)
        
        return self.maxCntNums
```

### Trie

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/Trie.jfif" width="350">

#### [208. Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/description/)
```python
class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for ch in word:
            cur = cur.children[ch]
        cur.is_word = True

    def search(self, word):
        cur = self.root
        for ch in word:
            cur = cur.children.get(ch)
            if cur is None:
                return False
        return cur.is_word

    def startsWith(self, prefix):
        cur = self.root
        for ch in prefix:
            cur = cur.children.get(ch)
            if cur is None:
                return False
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

#### [677. Map Sum Pairs](https://leetcode.com/problems/map-sum-pairs/description/)
```python
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.sum = 0  # Store the sum of values of all strings go through this node.

class MapSum:
    def __init__(self):
        self.trieRoot = TrieNode()
        self.map = defaultdict(int)

    def insert(self, key: str, val: int) -> None:
        # if the same key is inserted with different value, it should be updated
        diff = val - self.map[key] 
        curr = self.trieRoot
        for c in key:
            curr = curr.children[c]
            curr.sum += diff
        self.map[key] = val

    def sum(self, prefix: str) -> int:
        curr = self.trieRoot
        for c in prefix:
            if c not in curr.children: return 0
            curr = curr.children[c]
        return curr.sum

# Your MapSum object will be instantiated and called as such:
# mapSum = MapSum()
# mapSum.insert("apple", 3)
# mapSum.sum("ap");           // return 3 (apple = 3)
# mapSum.insert("app", 2)    
# mapSum.sum("ap");           // return 5 (apple + app = 3 + 2 = 5)
# mapSum.insert("apple", 5)    
# mapSum.sum("ap")           // return 7 (apple + app = 5 + 2 = 7)
```

## Stack and Queue
* [232. Implement Queue using Stacks](#232-Implement-Queue-using-Stacks)
* [225. Implement Stack using Queues](#225-Implement-Stack-using-Queues)
* [155. Min Stack](#155-Min-Stack)
* [20. Valid Parentheses](#20-Valid-Parentheses)
* [496. Next Greater Element I](#496-Next-Greater-Element-I)
* [739. Daily Temperatures](#739-Daily-Temperatures)
* [503. Next Greater Element II](#503-Next-Greater-Element-II)
* [1944. Number of Visible People in a Queue](#1944-Number-of-Visible-People-in-a-Queue)

#### [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
```python
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        self.s1.append(x)

    def pop(self):
        self.peek()
        return self.s2.pop()

    def peek(self):
        if not self.s2: # it's important that we dump only if s2 is not empty
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]        

    def empty(self):
        return not self.s1 and not self.s2


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

#### [225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)
```python
class MyStack:

    def __init__(self):
        self._queue = collections.deque()

    def push(self, x):
        q = self._queue
        q.append(x)
        for _ in range(len(q) - 1):
            q.append(q.popleft())
        
    def pop(self):
        return self._queue.popleft()

    def top(self):
        return self._queue[0]
    
    def empty(self):
        return not len(self._queue)

# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

#### [155. Min Stack](https://leetcode.com/problems/min-stack/description/)
```python
class MinStack(object):
    # e.g. [12,12], [30,12], [7,7], [6,6], [45,6], [2,2], [11,2]
    def __init__(self):
        self.stack = []
        
    def push(self, x):
        self.stack.append((x, min(self.getMin(), x))) 
           
    def pop(self):
        self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1][0]
        
    def getMin(self):
        if self.stack:
            return self.stack[-1][1]
        return float('inf') # depends on what interviewer asks for


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```
an alternative solution: use a min_tracker stack to store the previous min and count [approach 3](https://leetcode.com/problems/min-stack/solution/)

#### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        dic = {'}':'{', ']':'[', ')':'('}
        for char in s:
            if char in dic.values():
                stack.append(char)
            elif char in dic.keys():
                if stack == [] or dic[char] != stack.pop() :
                    return False
            else:
                return False
        
        return stack == []
```

#### [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
We use a stack to keep a decreasing sub-sequence, thus element on top, i.e. stack[-1], is the smallest value in the stack <br />
whenever we see a number x greater than stack[-1] we pop all elements less than x <br /> and then append x <br />
for all the popped ones, their next greater element is x <br />
For example [9, 8, 7, 3, 2, 1, 6] <br />
The stack will first contain [9, 8, 7, 3, 2, 1] and then we see 6 which is greater than 1 so we pop 1 2 3 whose next greater element should be 6 <br />
```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic = {}
        stack = deque([])
        for num in nums2:
            while stack and stack[-1] < num:
                dic[stack.pop()] = num
            stack.append(num)
 
        for i, x in enumerate(nums1):
            nums1[i] = dic.get(x, -1)
        
        return nums1
```

#### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/)
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        answer = [0] * n
        stack = []
        
        for curr_day, curr_temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < curr_temp:
                prev_day = stack.pop()
                answer[prev_day] = curr_day - prev_day
            stack.append(curr_day)
        
        return answer
```
e.g. [73, 74, 75, 71, 69, 72, 76, 73]. Iterating backwards, after 5 days we have: answer = [0, 0, 0, 2, 1, 1, 0, 0].  <br />
The next day to calculate is the day at index 2 with temperature 75.  <br />
First check the next day at index 3 - a temperature of 71, which is not warmer.  <br />
answer[3] = 2 tells us that the day at index 3 will see a warmer temperature on day 3 + 2 = 5.  <br />
A temperature warmer than 75 must also be warmer than 71 - so we should check temperatures[5] = 72 < 75.  <br />
Again, we know from answer[5] that we will not have a warmer temperature than 72 for 1 day.  <br />
Therefore, the next day to check is temperatures[5 + answer[5]] = temperatures[6] = 76, which is warmer - we found our day.
```python
# O(1) space
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        hottest = 0
        answer = [0] * n
        
        for curr_day in range(n - 1, -1, -1):
            curr_temp = temperatures[curr_day]
            if curr_temp >= hottest:
                hottest = curr_temp
                continue
            
            days = 1
            while temperatures[curr_day + days] <= curr_temp:  
                days += answer[curr_day + days]
            answer[curr_day] = days

        return answer
```

#### [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/description/)
```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [-1] * n
        stack = deque([])
        
        for idx in range(2 * n):
            num = nums[idx % n]
            while stack and nums[stack[-1]] < num:
                res[stack.pop()] = num
            if idx < n:
                stack.append(idx)

        return res
```

#### [1944. Number of Visible People in a Queue](https://leetcode.com/problems/number-of-visible-people-in-a-queue/)
```python
class Solution:
    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        res = [0] * len(heights)
        stack = [] # a decreasing mono stack
        for i, v in enumerate(heights):
            while stack and heights[stack[-1]] <= v:
                res[stack.pop()] += 1
            if stack:
                res[stack[-1]] += 1
                
            stack.append(i)
            
        return res
```


## HashMap
* [1. Two Sum](#1-Two-Sum)
* [217. Contains Duplicate](#217-Contains-Duplicate)
* [594. Longest Harmonious Subsequence](#594-Longest-Harmonious-Subsequence)
* [128. Longest Consecutive Sequence](#128-Longest-Consecutive-Sequence)


#### [1. Two Sum](https://leetcode.com/problems/two-sum/description/)
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i, val in enumerate(nums):
            if val in dic:
                return [dic[val], i]
            else:
                dic[target - val] = i
```

#### [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/description/)
```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        num_set = set()
        
        for num in nums:
            if num in num_set:
                return True
            else:
                num_set.add(num)
                
        return False
```
alternatives: sort it first and check for nums[i] and nums[i-1] with O(nlogn) time and O(1) space

#### [594. Longest Harmonious Subsequence](https://leetcode.com/problems/longest-harmonious-subsequence/description/)
```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        dic = defaultdict(int)
        
        count = 0
        for num in nums:
            dic[num] += 1
            
        for num in dic:
            if num + 1 in dic:
                count = max(count, dic[num] + dic[num + 1])
        
        return count
```

#### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/description/)
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        length = 0
        for x in numSet:
            if x - 1 in numSet:
                continue
                
            end = x + 1
            while end in numSet:
                end += 1
            length = max(length, end - x)
                
        return length
```


## Graph
* [785. Is Graph Bipartite](#785-Is-Graph-Bipartite)
* [207. Course Schedule](#207-Course-Schedule)
* [210. Course Schedule II](#210-Course-Schedule-II)
* [684. Redundant Connection](#684-Redundant-Connection)
* [1319. Number of Operations to Make Network Connected](#1319-Number-of-Operations-to-Make-Network-Connected)
* [721. Accounts Merge](#721-Accounts-Merge)


#### [785. Is Graph Bipartite](https://leetcode.com/problems/is-graph-bipartite/description/)
```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        color = {}
        for node in range(len(graph)):
            if node not in color:
                stack = [node]
                color[node] = 0
                while stack:
                    u = stack.pop()
                    for v in graph[u]:
                        if v not in color:
                            stack.append(v)
                            color[v] = color[u] ^ 1
                        elif color[v] == color[u]:
                            return False
        return True
```

#### [207. Course Schedule](https://leetcode.com/problems/course-schedule/description/)
```python
# DFS
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        prereq = [[] for _ in range(numCourses)]
        visit = [0 for _ in range(numCourses)] # 0 = unvisited
        
        for i, pre in prerequisites:
            prereq[i].append(pre)
            
        def dfs(i):
            if visit[i] == -1: # cycle detected as we visit i again
                return False
            
            if visit[i] == 1: # visited before, skip
                return True
            
            visit[i] = -1 # mark as visited
            for pre in prereq[i]:
                if not dfs(pre):
                    return False
                
            visit[i] = 1
            return True
        
        for i in range(numCourses):
            if not dfs(i):
                return False
            
        return True
```
```python
# BFS
import collections
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = [[] for _ in range(numCourses)] #{k: [v1, v2, ...]} k is prereq for [v1, v2, ...]
        in_degree = [0]*numCourses # number of prereq
        result = 0
        
        for i, pre in prerequisites:
            edges[pre].append(i)
            in_degree[i] += 1
        
        q = collections.deque([u for u in range(numCourses) if in_degree[u]==0]) # course with no prereq
        
        while q:
            u = q.popleft()
            result += 1
            for v in edges[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        
        return result == numCourses
```

#### [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/description/)
Topological Order
```python
# Solution 1: BFS
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj_list = [[] for _ in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]
        res = []
        
        for i, pre in prerequisites:
            adj_list[pre].append(i)
            indegree[i] = indegree[i] + 1

        zero_indegree_queue = deque([i for i in range(numCourses) if indegree[i] == 0])
        
        while zero_indegree_queue:
            i = zero_indegree_queue.popleft()
            res.append(i)

            # Reduce in-degree for all the neighbors
            for nei in adj_list[i]:
                indegree[nei] -= 1

                # Add neighbor to Q if in-degree becomes 0
                if indegree[nei] == 0:
                    zero_indegree_queue.append(nei)

        return res if len(res) == numCourses else []
```
graph illustration of DFS approach: [DFS](https://leetcode.com/problems/course-schedule-ii/solution/)
```python
# Solution 2: DFS - O(V + E) time and O(V + E) space
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        prereq = [[] for _ in range(numCourses)]
        visit = [0 for _ in range(numCourses)] # DAG detection 
        res = []
        
        for i, pre in prerequisites:
            prereq[i].append(pre) 
    
        def dfs(i):
            if visit[i] == -1:  # cycle detected as we visit i again
                return False
            
            if visit[i] == 1: # visited before and added to res already, skip
                return True
            
            visit[i] = -1 # mark as visited
            for pre in prereq[i]:
                if not dfs(pre):
                    return False
                
            visit[i] = 1
            res.append(i)
            return True
        
        for i in range(numCourses):
            if not dfs(i):
                return []
            
        return res
```

#### [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/description/)
DFS approach: We build the tree from scratch by adding edges [u, v] to adj_list. Before we add the edge, we do dfs to see if we can reach v from u through a path in existing graph. If we can, then adding [u,v] will form a cycle.
```python
# Solution 1: DFS - O(N^2) time and O(N) space 
# In the worst case, for every edge we include, we have to search every previously-occurring edge of the graph
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        
        def is_already_connected(x, y):
            if x in visited:
                return False
            
            if x == y:
                return True
            
            visited.add(x)
            for nei in adj_list[x]:
                if is_already_connected(nei, y):
                        return True
            return False
        
        adj_list = defaultdict(list)
        for x, y in edges:
            visited = set()
        
            if is_already_connected(x, y):
                return [x, y]
            
            adj_list[x].append(y)
            adj_list[y].append(x)
```
find(u) outputs a unique id so that two nodes have the same id if and only if they are in the same connected component. <br />
union(u, v) connects the components with id find(u) and find(v) together. If it already connected then return False, else return True.
```python
# Solution 2: Union Find
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1] * n
        
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x]) # Path compression
        return self.parent[x]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        
        if pu == pv: 
            return False  # u and v are already union
        
        if self.size[pu] < self.size[pv]: # Union by larger size
            self.parent[pu] = pv
            self.size[pv] += self.size[pu]
        else:
            self.parent[pv] = pu
            self.size[pu] += self.size[pv]
            
        return True

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        uf = UnionFind(n)
        for u, v in edges:
            if not uf.union(u-1, v-1): 
                return [u, v]
```
the self.size also tells us the size of each disjoint set. If we don't need to know that, we can also use "rank"
```python
self.rank = [0] * n

def union(self, u, v):
    pu, pv = self.find(u), self.find(v)

    if pu == pv: 
        return False  # u and v are already union

    if self.rank[pu] < self.rank[pv]:
        self.parent[pu] = pv
        self.rank[pv] += 1
    else:
        self.parent[pv] = pu
        self.rank[pu] += 1

    return True
    
# everything else the same
```

#### [1319. Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/)
The trick is, if we have enough cables, we don't need to worry about where we can get the cable from <br />
The number of operations we need = the number of connected networks - 1 <br />
Then the problem becomes, finding the number of connected components
```python
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n - 1: 
            return -1 # we need at least n - 1 cables to connect n nodes (like a tree)
        
        adj_list = [[] for _ in range(n)]
        for i, j in connections:
            adj_list[i].append(j)
            adj_list[j].append(i)
            
        seen = [0] * n
        def dfs(i):
            if seen[i]: 
                return 0
            
            seen[i] = 1
            
            for j in adj_list[i]: 
                dfs(j)
                
            return 1

        return sum(dfs(i) for i in range(n)) - 1
```

#### [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/)
```python
# Solution 1: DFS
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        visited_accounts = [False] * len(accounts)
        email_acc_map = defaultdict(list) # key=emails, value=account Id
        res = []

        for Id, acc in enumerate(accounts):
            for email in acc[1:]:
                email_acc_map[email].append(Id)
                
        def dfs(Id):           
            visited_accounts[Id] = True
            for email in accounts[Id][1:]:
                if email in seen:
                    continue
                    
                seen.add(email)
                for acc_Id in email_acc_map[email]:
                    if not visited_accounts[acc_Id]:
                        dfs(acc_Id)
                    
        for Id, acc in enumerate(accounts):
            if visited_accounts[Id]:
                continue
            seen = set()
            dfs(Id)
            res.append([acc[0]] + sorted(seen))
        return res
```
```python
# Solution 2: Union Find
class UF:
    def __init__(self, N):
        self.parents = list(range(N))
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        self.parents[pu] = pv
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        uf = UF(len(accounts))
        
        # Creat unions between indexes
        seen = {}
        for Id, emails in enumerate(accounts):
            for email in emails[1:]:
                if email in seen:
                    uf.union(Id, seen[email]) # union Ids
                seen[email] = Id
        
        ans = collections.defaultdict(list)
        for email, Id in seen.items():
            ans[uf.find(Id)].append(email)
        
        return [[accounts[i][0]] + sorted(emails) for i, emails in ans.items()]
```


## Sorting
* [215. Kth Largest Element in an Array](#215-Kth-Largest-Element-in-an-Array)
* [347. Top K Frequent Elements](#347-Top-K-Frequent-Elements)
* [451. Sort Characters By Frequency](#451-Sort-Characters-By-Frequency)
* [75. Sort Colors](#75-Sort-Colors)


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


## Greedy
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


## Divide and Conquer
* [241. Different Ways to Add Parentheses](#241-Different-Ways-to-Add-Parentheses)
* [96. Unique Binary Search Trees]([#96-Unique-Binary-Search-Trees)
* [95. Unique Binary Search Trees II](#95-Unique-Binary-Search-Trees-II)


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


## Binary Search
* [69. Sqrt x](#69-Sqrt-x)
* [744. Find Smallest Letter Greater Than Target](#744-Find-Smallest-Letter-Greater-Than-Target)
* [540. Single Element in a Sorted Array](#540-Single-Element-in-a-Sorted-Array)
* [278. First Bad Version](#278-First-Bad-Version)
* [153. Find Minimum in Rotated Sorted Array](#153-Find-Minimum-in-Rotated-Sorted-Array)
* [34. Find First and Last Position of Element in Sorted Array](#34-Find-First-and-Last-Position-of-Element-in-Sorted-Array)


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
in this case where r is assigned m, we need while loop of l < r instead of l <= r to avoid possibly infinite loop.

In some scenarios, you may see the loop condition as l < r instead of l <= r. This is usually when we are confident that the solution exists within the array and we do not want to check the same index multiple times.

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

## Search
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
	* [78. Subsets](#78-Subsets)
	* [90. Subsets II](#90-Subsets-II)
	* [77. Combinations](#77-Combinations)
	* [39. Combination Sum](#39-Combination-Sum)
	* [40. Combination Sum II](#40-Combination-Sum-II)
	* [216. Combination Sum III](#216-Combination-Sum-III)
	* [46. Permutations](#46-Permutations)
	* [47. Permutations II](#47-Permutations-II)
	* [131. Palindrome Partitioning](#131-Palindrome-Partitioning)
	* [267. Palindrome Permutation II](#267-Palindrome-Permutation-II)
	* [93. Restore IP Addresses](#93-Restore-IP-Addresses)
	* [79. Word Search](#79-Word-Search)
	* [257. Binary Tree Paths](#257-Binary-Tree-Paths)
	* [37. Sudoku Solver](#37-Sudoku-Solver)
	* [51. N Queens](#51-N-Queens)


### BFS

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
the execution of the backtracking is unfolded as a DFS traversal in a n-ary tree. The total number of steps during the backtracking would be the number of nodes in the tree.

Things to look out for during interviews:
- Always remember to define a base case(s) so that your recursion will end.
- Recursion implicitly uses a stack. Hence all recursive approaches can be rewritten iteratively using a stack. Beware of cases where the recursion level goes too deep and causes a stack overflow (the default limit in Python is 1000). Point this out to the interviewer if asked to implement a recursion.


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

#### [78. Subsets](https://leetcode.com/problems/subsets/)
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(index, path):
            combinations.append(path[:])
            
            for i in range(index, n):
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()
        
        n = len(nums)
        combinations = []
        backtrack(0, [])
            
        return combinations
```

#### [90. Subsets II](https://leetcode.com/problems/subsets-ii/)
```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def backtrack(index, path):
            combinations.append(path[:])
            
            for i in range(index, n):
                if i > index and nums[i] == nums[i - 1]:
                    continue
                    
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()
        
        n = len(nums)
        nums.sort()
        combinations = []
        backtrack(0, [])
            
        return combinations
```

#### [77. Combinations](https://leetcode.com/problems/combinations/)
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(index, path):
            if len(path) == k:
                combinations.append(path[:])
                return 
           
            for i in range(index, n):
                path.append(i + 1)
                backtrack(i + 1, path)
                path.pop()

        combinations = []
        backtrack(0, [])
	
        return combinations
```
		
#### [39. Combination Sum](https://leetcode.com/problems/combination-sum/)
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(index, path, remain):
            if remain == 0:
                combinations.append(path[:])
                return 
            elif remain < 0:
                return
           
            for i in range(index, n):
                path.append(candidates[i])
                backtrack(i, path, remain - candidates[i]) # note we start at i again, as we are 
							   # allowed to use candidate multiple times
                path.pop()

        n = len(candidates)
        combinations = []
        backtrack(0, [], target)
	
        return combinations
```

#### [40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(index, path, remain):
            if remain == 0:
                combinations.append(path[:])
                return 
            elif remain < 0:
                return
           
            for i in range(index, n):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                    
                path.append(candidates[i])
                backtrack(i + 1, path, remain - candidates[i])
                path.pop()

        n = len(candidates)
        candidates.sort()
        combinations = []
        backtrack(0, [], target)
	
        return combinations
```

#### [216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)
```python
class Solution:
    def combinationSum3(self, k: int, target: int) -> List[List[int]]:
        def backtrack(index, path, remain):
            if remain == 0 and len(path) == k:
                combinations.append(path[:])
                return 
            elif remain < 0 or len(path) == k:
                return
           
            for i in range(index, 9):
                path.append(i + 1)
                backtrack(i + 1, path, remain - (i + 1))
                path.pop()

        combinations = []
        backtrack(0, [], target)
	
        return combinations
```

#### [46. Permutations](https://leetcode.com/problems/permutations/)
<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/46.%20Permutations.png" width="700">

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        def backtrack(index):
            if index == n:
                ans.append(nums[:])
                
            for i in range(index, n):
                nums[index], nums[i] = nums[i], nums[index]
                backtrack(index + 1)
                nums[index], nums[i] = nums[i], nums[index]
                
        ans = []
        n = len(nums)
        backtrack(0)
        return ans
```

#### [47. Permutations II](https://leetcode.com/problems/permutations-ii/)
```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path, counter):
            if len(path) == len(nums):
                results.append(path[:])
                return

            for num in counter:
                if counter[num] > 0:
                    path.append(num)
                    counter[num] -= 1
                    backtrack(path, counter)
                    path.pop()
                    counter[num] += 1

        results = []
        backtrack([], Counter(nums))

        return results
```

#### [131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)
```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def backtracking(index, path):
            if index == n:
                res.append(path[:])
            
            for i in range(index, n):
                curr = s[index : i + 1] 
                
                if curr == curr[::-1]: # check palindrome
                    path.append(curr)
                    backtracking(i + 1, path)
                    path.pop()

        n = len(s)
        res = []
        backtracking(0, [])
        
        return res
```

#### [267. Palindrome Permutation II](https://leetcode.com/problems/palindrome-permutation-ii/)
```python
class Solution:
    def generatePalindromes(self, s: str) -> List[str]:
        counter = collections.Counter(s)
        mid = ''
        half = []
        for char, count in counter.items():
            q, r = divmod(count, 2)
            half += char * q
            
            if r == 1:
                if mid == '':
                    mid = char 
                else:
                    return [] # only one single char is acceptable
            
        def backtrack(path):
            if len(path) == n:
                cur = ''.join(path)
                ans.append(cur + mid + cur[::-1])
            else:
                for i in range(n):
                    if visited[i] or (i > 0 and half[i] == half[i-1] and not visited[i-1]):
                        continue
                    visited[i] = True
                    path.append(half[i])
                    backtrack(path)
                    visited[i] = False
                    path.pop()
                    
        ans = []
        n = len(half)
        visited = [False] * len(half)
        backtrack([])
        
        return ans
```

#### [93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)
```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        
        def backtrack(k, path, s):
            if k == 4:
                if len(s) == 0:
                    res.append('.'.join(path))
                else:
                    return
            
            for i in range(min(3, len(s))):
                if s[0] == '0' and i != 0:
                    break
                    
                part = s[0 : i + 1]

                print(s, part)
                if int(part) <= 255:
                    path.append(part)
                    backtrack(k + 1, path, s[i + 1:])
                    path.pop()
                        
        res = []
        backtrack(0, [], s)
        
        return res
```
#### [79. Word Search](https://leetcode.com/problems/word-search/)
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:

        def backtrack(row, col, s):
            if len(s) == 0:
                return True

            if row < 0 or row == ROWS or col < 0 or col == COLS or board[row][col] != s[0]:
                return False

            ret = False
            board[row][col] = '#'
            
            for d1, d2 in directions:
                ret = backtrack(row + d1, col + d2, s[1:])
                if ret: 
                    break # no clean-up if we do "return True" here (sudden-death return)

            # revert the change
            board[row][col] = s[0]

            return ret
        
        ROWS = len(board)
        COLS = len(board[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for row in range(ROWS):
            for col in range(COLS):
                if backtrack(row, col, word):
                    return True

        return False
```

#### [257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []
        
        def backtrack(root, path):

            path.append(str(root.val))

            if root.left:
                backtrack(root.left, path)
                path.pop()

            if root.right:
                backtrack(root.right, path)
                path.pop()

            if root.left is None and root.right is None:
                res.append('->'.join(path))
                
        
        res = []
        backtrack(root, [])
        
        return res
```

#### [37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/submissions/)
```python
from collections import defaultdict
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
                
        def is_valid(r, c, v):
            box_id = (r // 3) * 3 + c // 3
            return v not in rows[r] and v not in cols[c] and v not in boxes[box_id]


        def backtrack(r, c):
            if c == n:
                if r == n - 1:
                    return True
                else:
                    c = 0
                    r += 1

            if board[r][c] != '.':
                return backtrack(r, c + 1)

            box_id = (r // 3) * 3 + c // 3
            for v in range(1, 10):
                if not is_valid(r, c, v):
                    continue

                board[r][c] = str(v)
                rows[r].add(v)
                cols[c].add(v)
                boxes[box_id].add(v)

                if backtrack(r, c + 1):
                    return True

                # backtrack
                board[r][c] = '.'
                rows[r].remove(v)
                cols[c].remove(v)
                boxes[box_id].remove(v)

            return False

        
        n = len(board)
        rows, cols, boxes = defaultdict(set), defaultdict(set), defaultdict(set)

        for r in range(n):
            for c in range(n):
                if board[r][c] == '.':
                    continue

                v = int(board[r][c])
                rows[r].add(v)
                cols[c].add(v)
                boxes[(r // 3) * 3 + c // 3].add(v)
                
        backtrack(0, 0)
```

#### [51. N Queens](https://leetcode.com/problems/n-queens/description/)
```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        
        def create_board(board):
            res = []
            for row in board:
                res.append("".join(row))
            return res
        
        def backtrack(row, diagonals, anti_diagonals, cols, board):
            if row == n:
                ans.append(create_board(board))
                return

            for col in range(n):
                curr_diagonal = row - col
                curr_anti_diagonal = row + col
                
                if (col in cols or curr_diagonal in diagonals or curr_anti_diagonal in anti_diagonals):
                    continue

                cols.add(col)
                diagonals.add(curr_diagonal)
                anti_diagonals.add(curr_anti_diagonal)
                board[row][col] = "Q"

                backtrack(row + 1, diagonals, anti_diagonals, cols, board)

                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)
                board[row][col] = "."

        ans = []
        empty_board = [["."] * n for _ in range(n)]
        backtrack(0, set(), set(), set(), empty_board)
        
        return ans
```

## Dynamic Programming
* [Fibonacci](#Fibonacci)
	* [70. Climbing Stairs](#70-Climbing-Stairs)
	* [198. House Robber](#198-House-Robber)
	* [213. House Robber II](#213-House-Robber-II)
	* [Mail Misalignment](#Mail-Misalignment)
	* [Cow](#Cow)
* [Matrix Path](#Matrix-Path) 
	* [64. Minimum Path Sum](#64-Minimum-Path-Sum)
	* [62. Unique Paths](#62-Unique-Paths)
* [Range](#Range)
	* [303. Range Sum Query](#303-Range-Sum-Query)
	* [413. Arithmetic Slices](#413-Arithmetic-Slices)
* [Breakdown Number](#Breakdown-Number)
	* [343. Integer Break](#343-Integer-Break)
	* [279. Perfect Squares dup](#279-Perfect-Squares-dup) 
	* [91. Decode Ways](#91-Decode-Ways)
* [Subsequence](#Subsequence)
	* [300. Longest Increasing Subsequence](#300-Longest-Increasing-Subsequence)
	* [646. Maximum Length of Pair Chain](#646-Maximum-Length-of-Pair-Chain)
	* [376. Wiggle Subsequence](#376-Wiggle-Subsequence)
	* [1143. Longest Common Subsequence](#1143-Longest-Common-Subsequence)
* [01 Bag](#01-Bag)
	* [416. Partition Equal Subset Sum](#416-Partition-Equal-Subset-Sum)
	* [494. Target Sum](#494-Target-Sum)
	* [474. Ones and Zeroes](#474-Ones-and-Zeroes)
	* [322. Coin Change](#322-Coin-Change)
	* [518. Coin Change 2](#518-Coin-Change-2)
	* [377. Combination Sum IV](#377-Combination-Sum-IV)
	* [139. Word Break](#139-Word-Break)
* [Stock Trade](#Stock-Trade)
	* [309. Best Time to Buy and Sell Stock with Cooldown](#309-Best-Time-to-Buy-and-Sell-Stock-with-Cooldown)
	* [714. Best Time to Buy and Sell Stock with Transaction Fee](#714-Best-Time-to-Buy-and-Sell-Stock-with-Transaction-Fee)
	* [123. Best Time to Buy and Sell Stock III](#123-Best-Time-to-Buy-and-Sell-Stock-III)
	* [188. Best Time to Buy and Sell Stock IV](#188-Best-Time-to-Buy-and-Sell-Stock-IV)
* [String Operations](#String-Operations)
	* [583. Delete Operation for Two Strings](#583-Delete-Operation-for-Two-Strings)
	* [72. Edit Distance](#72-Edit-Distance)
	* [650. 2 Keys Keyboard](#650-2-Keys-Keyboard)


Dynamic Programming is recursive approach with memorization

Example: Bowling
Given n pins 0, 1, ..., n-1, where pin i has value V_i. We get V_i point by hitting 1 pin i, and get V_i * V_i+1 by hitting 2 pins i and i + 1. We want to get max score.

**SORTBT**
- **S**ubproblem: B(i) = max score possible starting with pin i, i + 1, ..., n-1
- **O**riginal: B(0)
- **R**elate: B(i) = max{B(i+1), B(i+1) + V_i, B(i+2) + V_i*V_i+1}
- **T**opological order: decreasing i, i.e. for i = n, n-1, ...., 0
- **B**ase: B(n) = 0
- **T**ime: O(n)

Bottom up DP implementation
```python
base     B(n) = 0
topo     for i = n, n-1, ..., 0:
relate      B(i) = max{B(i+1), B(i+1) + V_i, B(i+2) + V_i*V_i+1}
original    return B(0)
```	
Good subproblem: 
- prefixes x[:i] O(n)
- suffixes x[i:] O(n)
- substrings x[i:j] O(n^2)

### Fibonacci
#### [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/)
```python
# Solution 1: bottom up DP (constant space)
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        
        first, second = 1, 2
        
        for i in range(3, n + 1):
            tmp = first + second
            first = second
            second = tmp
            
        return second
	
# Solution 2: bottom up DP
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = {}
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

# Solution 3: top down DP
class Solution:
    def climbStairs(self, n: int) -> int:
        def climb(n):
            if n not in dic:
                dic[n] = climb(n-1) + climb(n-2)

            return dic[n]  
    
        dic = {1:1, 2:2}
        return climb(n)
```

#### [198. House Robber](https://leetcode.com/problems/house-robber/)
```python
# Solution 1: DP (constant space)
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        N = len(nums)
        rob_next_next = 0
        rob_next = nums[N - 1]
        
        for i in range(N - 2, -1, -1):
            maxRobbedAmount = max(rob_next, rob_next_next + nums[i])
            rob_next_next = rob_next
            rob_next = maxRobbedAmount
            
        return rob_next
            
# Solution 2: DP 
class Solution:
    def rob(self, nums: List[int]) -> int:
        # subproblem: suffix dp[i:]
        # topo. order: N, N-1, ..., 0
        
        if not nums:
            return 0
        
        N = len(nums)
        maxRobbedAmount = [None for _ in range(N + 1)]
        
        # Base case
        maxRobbedAmount[N], maxRobbedAmount[N - 1] = 0, nums[N - 1]
        
        # relate
        for i in range(N - 2, -1, -1):
            maxRobbedAmount[i] = max(maxRobbedAmount[i + 1], maxRobbedAmount[i + 2] + nums[i])
            
        # original
        return maxRobbedAmount[0] 
```

#### [213. House Robber II](https://leetcode.com/problems/house-robber-ii/)
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
          
        if not nums or len(nums) == 0:
            return 0
        
        if len(nums) == 1:
            return nums[0]
        
        def rob_helper(nums):
            N = len(nums)
            rob_next_next = 0
            rob_next = nums[N - 1]

            for i in range(N - 2, -1, -1):
                maxRobbedAmount = max(rob_next, rob_next_next + nums[i])
                rob_next_next = rob_next
                rob_next = maxRobbedAmount

            return rob_next
        
        return max(rob_helper(nums[1:]), rob_helper(nums[:-1]))
```

#### Mail Misalignment
有N个信和信封, 它们被打乱, 求错误装信方式的数量。

思路：定义一个数组 dp 存储错误方式数量，dp[i] 表示前 i 个信和信封的错误方式数量。假设第 i 个信装到第 j 个信封里面，而第 j 个信装到第 k 个信封里面。根据 i 和 k 是否相等，有两种情况：

- i == k，交换 i 和 j 的信后，它们的信和信封在正确的位置，但是其余 i-2 封信有 dp[i-2] 种错误装信的方式。由于 j 有 i-1 种取值，因此共有 (i-1)*dp[i-2] 种错误装信方式。
- i != k，交换 i 和 j 的信后，第 i 个信和信封在正确的位置，其余 i-1 封信有 dp[i-1] 种错误装信方式。由于 j 有 i-1 种取值，因此共有 (i-1)*dp[i-1] 种错误装信方式。

综上所述，错误装信数量方式数量为：dp[i] = (i-1)*dp[i-2] + (i-1)*dp[i-1]

```python
def MailMisalignment(int n):
    if n == 0 or n == 1:
    	return 0
	
    dp = [None] * (n+1)
    dp[0] = 0
    dp[1] = 0
    dp[2] = 1
    
    for i in range(3, n):
        dp[i] = (i-1)*dp[i-2] + (i-1)*dp[i-1]
	
    return dp[n]
```

#### Cow
假设农场中成熟的母牛每年都会生 1 头小母牛，并且永远不会死。第一年有 1 只小母牛，从第二年开始，母牛开始生小母牛。每只小母牛 3 年之后成熟又可以生小母牛。给定整数 N，求 N 年后牛的数量。

思路：dp数组存储每年成熟小母女的数量。因为只有成熟的母女才会继续生产牛。例：1，2，3，4，6 ... 即第N年后牛的数量等于第N - 1年牛的数量，加上第N - 3年成熟小母牛下的小母牛(3年成熟)。

状态转移方程：dp[i] = dp[i - 1] + dp[i - 3], n > 3
```python
def cow(int n):
    if i < 4:
    	return n
	
    dp = [None] * (n + 1)

    for i in range(5):
        dp[i] = i

    for i in range(5, n+1):
        dp[i] = dp[i - 1] + dp[i - 3]
   
    return dp[n]
```

### Matrix Path

#### [64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # base case
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
            
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
                
        return grid[-1][-1]
```

#### [62. Unique Paths](https://leetcode.com/problems/unique-paths/description/)
```python
# Solution 1: DP
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        d = [[1] * n for _ in range(m)]

        for r in range(1, m):
            for c in range(1, n):
                d[r][c] = d[r - 1][c] + d[r][c - 1]

        return d[-1][-1]
```
```python
# Solution 2: Math
from math import factorial
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return factorial(m + n - 2) // (factorial(n - 1) * factorial(m - 1))
```

### Range

#### [303. Range Sum Query](https://leetcode.com/problems/range-sum-query-immutable/)
```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.preSum = nums
        for i in range(len(nums)-1):
            self.preSum[i+1] += self.preSum[i]
            
    def sumRange(self, left: int, right: int) -> int:
        if left == 0: 
            return self.preSum[right]
        
        return self.preSum[right] - self.preSum[left-1]
```

#### [413. Arithmetic Slices](https://leetcode.com/problems/arithmetic-slices/)
use dp[i] to store the number of arithmetic slices possible in the range (k,i), where k refers to the minimum index such that the range (k,i) constitutes a valid arithmetic slice, and (k,i) is not a part of any range (k,j) where j<i. 

consider the range (0, i-1), constituted by the elements a(0), ..., a(i-1). Let's denote the number of arithmetic slices as dp[i-1]. 

Now, add a new element a(i) with the same difference as previous ones. Note that a(1), ..., a(i) can be mapped perfectly to a(0), ..., a(i-1). Thus a(1), ..., a(i) has the same number of arithmetic slices as a(0), ..., a(i-1), which is dp[i-1]. 

And there is one more sequence: a(0), ..., a(i). Therefore, dp[i] = dp[i-1] + 1. And the total number of arithmetic slices in a(0), ..., a(i) is dp[i-1] + dp[i] = dp[i-1] + (dp[i-1] + 1)
```python
# Solution 1: DP
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        ans = 0
        for i in range(2, n):
            if nums[i-1] - nums[i-2] == nums[i] - nums[i-1]:
                dp[i] = dp[i-1] + 1
            ans += dp[i]
        return ans
    
# Solution 2: DP with constant space        
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        dp, dpPrev = 0, 0
        ans = 0
        for i in range(2, n):
            if nums[i-1] - nums[i-2] == nums[i] - nums[i-1]:
                dp = dpPrev + 1
            ans += dp
            dpPrev = dp
            dp = 0
        return ans
```

### Breakdown Number

#### [343. Integer Break](https://leetcode.com/problems/integer-break/description/)
write i as: i = j + S where S = i - j corresponds to either one number or a sum of two or more numbers
1. S is a single number: dp[i] = j * (i - j)
2. S is a sum of at least 2 numbers: dp[i] = j * dp[i - j]
```python
# Solution 1: DP
class Solution:
    def integerBreak(self, n: int) -> int:
        
        dp = [0] * (n + 1);
        dp[1] = 1
        
        for i in range(2, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], max(j * dp[i - j], j * (i - j)))
    
        return dp[n]
```
```python
# Solution 2: Math
class Solution:
    def integerBreak(self, n: int) -> int:
        if n==2:
            return 1
        
        if n==3: 
            return 2
        
        product = 1
        while n > 4:
            product *= 3
            n -= 3
        
        product *= n;
        
        return product;
```

#### [279. Perfect Squares dup](https://leetcode.com/problems/perfect-squares/)
```python
# Solution 1: DP - O(N*√N) time and O(N) space
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
```python
# Solution 2: BFS
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

#### [91. Decode Ways](https://leetcode.com/problems/decode-ways/description/)
```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s:
            return 0

        n = len(s)
        dp = [0] * (n + 1)

        dp[0] = 1 
        dp[1] = 0 if s[0] == "0" else 1

        for i in range(2, len(s) + 1): 

            if 0 < int(s[i-1:i]) <= 9:
                dp[i] += dp[i - 1]

            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i - 2]
                
        return dp[len(s)]
```

### Subsequence

#### [300. Longest Increasing Subsequence](#300-Longest-Increasing-Subsequence)
let dp[i] represents the length of the longest increasing subsequence that **ends with nums[i]**. Then, dp[i] = max(dp[j] + 1) for all j where nums[j] < nums[i] and j < i.
```python
# Solution 1: DP
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
```

Initialize an array sub which contains the first element of nums.
Iterate through the input, starting from the second element. For each element num: If num is greater than any element in sub, then add num to sub. Otherwise, iterate through sub and find the first element that is greater than or equal to num. Replace that element with num.
Return the length of sub.
```python
# Solution 2: math
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def binarySearch(nums, key):
            l = 0
            r = len(nums) - 1

            while l < r:
                m = l + (r - l) // 2

                if nums[m] == key:
                    return m
                elif nums[m] > key:
                    r = m
                else:
                    l = m + 1

            return l
        
        sub = [nums[0]]
        for num in nums[1:]:
            if num > sub[-1]:
                sub.append(num)
            else:
                idx = binarySearch(sub, num)
                sub[idx] = num
        
        return len(sub)
```

#### [646. Maximum Length of Pair Chain](https://leetcode.com/problems/maximum-length-of-pair-chain/)
```python
# Solution 1: interval
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        N = len(pairs)
        ans = 0
        pairs.sort(key = lambda x: x[1])

        prevEnd = float('-inf')
        for head, tail in pairs:
            if head > prevEnd:
                prevEnd = tail
                ans += 1
                
        return ans
```
```python
# Solution 2: DP
class Solution(object):
    def findLongestChain(self, pairs):
        pairs.sort()
        dp = [1] * len(pairs)

        for j in range(len(pairs)):
            for i in range(j):
                if pairs[i][1] < pairs[j][0]:
                    dp[j] = max(dp[j], dp[i] + 1)

        return max(dp)
```

#### [376. Wiggle Subsequence](https://leetcode.com/problems/wiggle-subsequence/)
```python
# Solution: greedy
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n < 2:
            return len(nums)

        length = 1
        sign = 0
        for i in range(1, n):
            if nums[i] < nums[i-1] and sign != -1: # peak
                sign = -1
                length += 1
            elif nums[i] > nums[i-1] and sign != 1: # valley
                sign = 1
                length += 1
       
        return length 
```

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/376.%20Wiggle%20Subsequence.png" width="700">

```python
# Solution: space optimized DP
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n < 2:
            return len(nums)
        
        down, up = 1, 1
        
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                up = down + 1
            elif nums[i] < nums[i - 1]:
                down = up + 1
                
        return max(down, up)

# Solution: DP
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n < 2:
            return len(nums)
        
        up = [0] * n
        down = [0] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    up[i] = max(up[i], down[j] + 1)
                elif  nums[i] < nums[j]:
                    down[i] = max(down[i], up[j] + 1)
                    
        return 1 + max(down[n - 1], up[n - 1])
```

#### [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/1143.%20Longest%20Common%20Subsequence.png" width="350">

```python
# Solution: DP with space optimization
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        if len(text2) < len(text1):
            text1, text2 = text2, text1
   
        n1 = len(text1)
        n2 = len(text2)

        previous = [0] * (n1 + 1)
        current = [0] * (n1 + 1)
        
        # fix the suffix of text 2 (col), and iterate up each letter in text 1
        for col in range(n2 - 1, -1, -1):
            for row in range(n1 - 1, -1, -1):
                if text2[col] == text1[row]:
                    current[row] = 1 + previous[row + 1] # previous is the "t" col in graph
		    					 # previous[row + 1] is the yellow "2"
							 # current is the "a" col in graph
							 # curret[row] is the green "3"
                else:
                    current[row] = max(previous[row], current[row + 1])
                    
            previous, current = current, previous
        
        return previous[0]

# Solution: DP
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1)
        n2 = len(text2)
        dp_grid = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        
        for col in range(n2 - 1, -1, -1):
            for row in range(n1 - 1, -1, -1):
                if text2[col] == text1[row]:
                    dp_grid[row][col] = 1 + dp_grid[row + 1][col + 1]
                else:
                    dp_grid[row][col] = max(dp_grid[row + 1][col], dp_grid[row][col + 1])
        
        return dp_grid[0][0]
```

### 01-Bag

有一个容量为 N 的背包，要用这个背包装下物品的价值最大，这些物品有两个属性：体积 w 和价值 v。

定义一个二维数组 dp 存储最大价值，其中 dp[i][j] 表示前 i 件物品体积不超过 j 的情况下能达到的最大价值。设第 i 件物品体积为 w，价值为 v，根据第 i 件物品是否添加到背包中，可以分两种情况讨论：

- 第 i 件物品没添加到背包，总体积不超过 j 的前 i 件物品的最大价值就是总体积不超过 j 的前 i-1 件物品的最大价值，dp[i][j] = dp[i-1][j]。
- 第 i 件物品添加到背包中，dp[i][j] = dp[i-1][j-w] + v。
第 i 件物品可添加也可以不添加，取决于哪种情况下最大价值更大。因此，0-1 背包的状态转移方程为：
```java
//W 为背包总体积
//N 为物品数量
//weights 数组存储 N 个物品的重量
//values 数组存储 N 个物品的价值
public int knapsack(int W, int N, int[] weights, int[] values) {
    int[][] dp = new int[N + 1][W + 1];
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = 1; j <= W; j++) {
            if (j >= w) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - w] + v);
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[N][W];
}
```

01背包的space optimization
在程序实现时可以对 0-1 背包做优化。观察状态转移方程可以知道，前 i 件物品的状态仅与前 i-1 件物品的状态有关，因此可以将 dp 定义为一维数组，其中 dp[j] 既可以表示 dp[i-1][j] 也可以表示 dp[i][j]。注意，要先计算 dp[i][j] 再计算 dp[i][j-w]，在程序实现时需要按倒序来循环求解。
```java
public int knapsack(int W, int N, int[] weights, int[] values) {
    int[] dp = new int[W + 1];
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = W; j >= 1; j--) {
            if (j >= w) {
                dp[j] = Math.max(dp[j], dp[j - w] + v);
            }
        }
    }
    return dp[W];
}
```

Variants:
- 完全背包：物品数量为无限个
- 多重背包：物品数量有限制
- 多维费用背包：物品不仅有重量，还有体积，同时考虑这两种限制
- 其它：物品之间相互约束或者依赖

#### [416. Partition Equal Subset Sum](#416-Partition-Equal-Subset-Sum)
思路：背包大小为 total_sum//2 的0-1背包问题

dp[i][j] = true if the sum j can be formed by elements in subset {nums[0], ..., nums[i]}, otherwise dp[i][j] = false
dp[i][j] = true in two cases:
1. sum j can be formed without including nums[i], i.e., dp[i-1][j] == true
2. sum j can be formed by including nums[i], i.e., dp[i−1][j−nums[i]]==true

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/416.%20Partition%20Equal%20Subset%20Sum.png" width="500">

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        dp[i][j] = true if the sum j can be formed by elements in 
        subset {nums[0], ..., nums[i]}
        '''
        total_sum = sum(nums)

        if total_sum % 2 != 0:
            return False
        
        subset_sum = total_sum // 2
        n = len(nums)

        dp = [[False] * (subset_sum + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            curr = nums[i - 1]
            for j in range(subset_sum + 1):
                if j < curr:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - curr]
        return dp[n][subset_sum]
	
# DP with constant space
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        subset_sum = total_sum // 2

        # construct a dp table of size (subset_sum + 1)
        dp = [False] * (subset_sum + 1)
        dp[0] = True
        for num in nums:
            for j in range(subset_sum, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]

        return dp[subset_sum]
```

#### [494. Target Sum](https://leetcode.com/problems/target-sum/description/)
思路：背包大小为 target + sum(nums)//2 的0-1背包问题
```
                  sum(P) - sum(N) = target
sum(P) + sum(N) + sum(P) - sum(N) = target + sum(P) + sum(N)
                       2 * sum(P) = target + sum(nums)
		       	   sum[P] = [target + sum(nums)] // 2
```

dp[i][j] is the number of ways to make the subset_sum = j using elements in subset {nums[0], ..., nums[i]}       
```python
# DP
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        
        if (total_sum + target) % 2 != 0 or total_sum < target:
            return 0
        
        subset_sum = (total_sum + target) // 2
        n = len(nums)
        
        dp = defaultdict(lambda: defaultdict(int)) # target could be negative, can't use array
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(subset_sum + 1):   
	    	if j < curr:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - num]
                
        return dp[n][subset_sum]
    
# space optimization
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        
        if (total_sum + target) % 2 != 0 or total_sum < target:
            return 0
        
        subset_sum = (total_sum + target) // 2

        dp = defaultdict(int)
        dp[0] = 1
        for num in nums:
            for j in range(subset_sum, num - 1, -1):
                dp[j] = dp[j] + dp[j - num]
                
        return dp[subset_sum]
```
```python
# DFS
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        def findTarget(i, target):
            if (i, target) not in cache:
                r = 0
                
                if i == len(nums):
                    if target == 0:
                        r = 1
                else:
                    r = findTarget(i + 1, target - nums[i]) + findTarget(i + 1, target + nums[i])
                cache[(i, target)] = r
                
            return cache[(i, target)]
        
        cache = {}
        return findTarget(0, target)
```
```python
# BFS
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        count = defaultdict(int)
        count[0] = 1
        for x in nums:
            step = defaultdict(int)
            for y in count:
                step[y + x] += count[y]
                step[y - x] += count[y]
            count = step

        return count[target]
```

#### [474. Ones and Zeroes](https://leetcode.com/problems/ones-and-zeroes/description/)
思路：多维费用的 0-1 背包问题，有两个背包大小，0 的数量和 1 的数量。
```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
              
        for s in strs:
            zeros, ones = 0, 0
            
            for c in s:
                if c == '0':
                    zeros += 1
                else:
                    ones += 1
              
            for i in range(m, zeros - 1, -1):
                  for j in range(n, ones - 1, -1):
                        dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
              
        return dp[m][n]
```

#### [322. Coin Change](https://leetcode.com/problems/coin-change/description/)
思路：完全背包问题, 只需要将 0-1 背包的逆序遍历 dp 数组改为正序遍历即可。
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[float('inf')] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            coin = coins[i - 1]
            for j in range(amount + 1):
                if j < coin:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - coin] + 1)
            
        return dp[n][amount] if dp[n][amount] != float('inf') else -1 
	
# space optimization
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
        return dp[amount] if dp[amount] != float('inf') else -1 
```

#### [518. Coin Change 2](https://leetcode.com/problems/coin-change-2/description/)
思路：完全背包问题
```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            coin = coins[i - 1]
            
            for j in range(amount + 1):
                if j < coin:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coin]
            
        return dp[n][amount]

# space optimization
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
	
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = dp[i] + dp[i - coin]
                
        return dp[amount]
```

#### [377. Combination Sum IV](377. Combination Sum IV)
```python
class Solution:
    def combinationSum4(self, nums: List[int], amount: int) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1

        for i in range(amount + 1):
            for num in nums:
                if i >= num:
                    dp[i] = dp[i] + dp[i - num]
                    
        return dp[amount]
```

#### [139. Word Break](https://leetcode.com/problems/word-break/)
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for w in wordDict:
                if dp[i-len(w)] and s[i-len(w):i] == w:
                    dp[i] = True
                    break
        return dp[n]
```

### Stock Trade

#### [309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/)
<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/309.%20Best%20Time%20to%20Buy%20and%20Sell%20Stock%20with%20Cooldown.png" width="500">

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sold, held, reset = float('-inf'), float('-inf'), 0

        for price in prices:
            pre_sold = sold
            sold = held + price
            held = max(held, reset - price)
            reset = max(reset, pre_sold)

        return max(sold, reset)
```

#### [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)
If I am holding a share after today, then either 
1. I am just continuing holding the share I had yesterday, or 
2. I held no share yesterday, but bought in one share today
```
hold = max(hold, not_hold - prices[i])
```
If I am not holding a share after today, then either
1. I did not hold a share yesterday, or 
2. I held a share yesterday but I decided to sell it out today
```
not_hold = max(not_hold, hold + prices[i] - fee)
```
note that we can calculate "not_hold" first without using temporary variables because selling and buying on the same day can't be better than just continuing to hold the stock
```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        not_hold, hold = 0, -prices[0]
        
        for i in range(1, len(prices)):
            not_hold = max(not_hold, hold + prices[i] - fee)
            hold = max(hold, not_hold - prices[i])
            
        return not_hold
```

#### [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/)

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/123.%20Best%20Time%20to%20Buy%20and%20Sell%20Stock%20III.png" width="700">

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        s1 = -prices[0]
        s2 = float('-inf')
        s3 = float('-inf')
        s4 = float('-inf')

        for i in range(1, len(prices)):           
            s1 = max(s1, - prices[i])
            s2 = max(s2, s1 + prices[i])
            s3 = max(s3, s2 - prices[i])
            s4 = max(s4, s3 + prices[i])
            print(prices[i], [s1, s2, s3, s4])
        return max(0, s4)
```
see Leetcode-188 below for generalization for k transactions

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        t1_cost, t2_cost = float('inf'), float('inf')
        t1_profit, t2_profit = 0, 0

        for price in prices:
            t1_cost = min(t1_cost, price)
            t1_profit = max(t1_profit, price - t1_cost)
            
            # reinvest the gained profit in the second transaction
            t2_cost = min(t2_cost, price - t1_profit)
            t2_profit = max(t2_profit, price - t2_cost)

        return t2_profit
```
see Leetcode-188 below for generalization for k transactions

#### [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if k==0:
            return 0

        if 2*k >= len(prices): 
            return sum(max(0, prices[i]-prices[i-1]) for i in range(1, len(prices)))
    
    
        states = [0] + [-float('inf') for i in range(2*k)]
        states[1] = -prices[0]

        for i in range(1, len(prices)):
            for j in range(k):
                states[2*j + 1] = max(states[2*j + 1], states[2*j] - prices[i])
                states[2*j + 2] = max(states[2*j + 2], states[2*j + 1] + prices[i])

        return max(0, states[-1])
```


```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if k == 0:
            return 0
        
        if 2*k >= len(prices): 
            return sum(max(0, prices[i]-prices[i-1]) for i in range(1, len(prices)))
    
        profit = [0] * (k+1)
        cost = [float('inf')] * (k+1)

        profit[0] = 0
        
        for price in prices:
            for i in range(1, k + 1):
                cost[i] = min(cost[i], price - profit[i - 1])
                profit[i] = max(profit[i], price - cost[i])

        return profit[k]
```

Dynamic Programming approach (detailed version of solution #1)

dp[day_number][used_transaction_number][stock_holding_status]. The value of dp[i][j][l] represents the best profit we can have at the end of the i-th day, with j remaining transactions to make and l stocks.

- Keep holding the stock: dp[i][j][1] = dp[i-1][j][1]
- Keep not holding the stock: dp[i][j][0] = dp[i-1][j][0]
- Buying, when j>0: dp[i][j][1] = dp[i-1][j-1][0] - prices[i]
- Selling: dp[i][j][0] = dp[i-1][j][1] + prices[i]

We can combine they together to find the maximum profit:
- dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
- dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)

        if k==0:
            return 0

        if 2*k >= len(prices): 
            return sum(max(0, prices[i]-prices[i-1]) for i in range(1, len(prices)))
    
        dp = [[[float('-inf')]*2 for _ in range(k + 1)] for _ in range(n)]
	
	dp[0][0][0] = 0
        dp[0][1][1] = -prices[0]

        for i in range(1, n):
            for j in range(k + 1):
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                if j > 0:
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])

        res = max(dp[n-1][j][0] for j in range(k + 1))
        
        return res
```

### String Operations

#### [583. Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/)
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        '''
        this problem is equivalent to longest common substring
        '''
        m = len(word1)
        n = len(word2)
        
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return m + n - 2 * dp[m][n]
```

#### [72. Edit Distance](#72-Edit-Distance)
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        
        # if one of the strings is empty
        if n * m == 0:
            return n + m
        
        d = [ [0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if word1[i - 1] == word2[j - 1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[n][m]
```

#### [650. 2 Keys Keyboard](https://leetcode.com/problems/2-keys-keyboard/)
```python
# DP
class Solution:
    def minSteps(self, n: int) -> int:
        dp = [0] * (n + 1)

        for i in range(2, n + 1):
            dp[i] = i # the possible maximum
            for j in range(1, i): # j is the existing 'A'
                if i % j == 0:
                    dp[i] = min(dp[i], dp[j] + (i//j))
                
        return dp[n]
```

We can break our moves into groups of (copy, paste, ..., paste). Let C denote copying and P denote pasting. 
Then for example, in the sequence of moves CPPCPPPPCP, the groups would be [CPP][CPPPP][CP].

Say these groups have lengths g_1, g_2, .... After parsing the first group, there are g_1 'A's. After parsing the second group, there are g_1 * g_2 'A's, and so on.
At the end, there are g_1 * g_2 * ... * g_n 'A's.

We want exactly N = g_1 * g_2 * ... * g_n. If any of the g_i are composite, say g_i = p * q, then we can split this group into two groups (the first of which has one copy followed by p-1 pastes, while the second group having one copy and q-1 pastes).
```python
class Solution:
    def minSteps(self, n: int) -> int:
        ans = 0
        d = 2
        
        while n > 1:
            while n % d == 0:
                ans += d
                n /= d
            d += 1
            
        return ans
```


## Math
* [204. Count Primes](#204-Count-Primes)
* [Greatest Common Divisor](#Greatest-Common-Divisor)
* [Least Common Multiple](#Least-Common-Multiple)
* [172. Factorial Trailing Zeroes](#172-Factorial-Trailing-Zeroes)
* [462. Minimum Moves to Equal Array Elements II](#462-Minimum-Moves-to-Equal-Array-Elements-II)
* [169. Majority Element](#169-Majority-Element)
* [367. Valid Perfect Square](#367-Valid-Perfect-Square)
* [628. Maximum Product of Three Numbers](#628-Maximum-Product-of-Three-Numbers)


#### [204. Count Primes](https://leetcode.com/problems/count-primes/)
```python
# Solution 1
class Solution:
    def countPrimes(self, n: int) -> int:
        if n <= 2:
            return 0
        
        primes = [False] * 2 + [True] * (n - 2)
        
        for i in range(2, int(sqrt(n)) + 1):
            if (primes[i] == True):
                for p in range(i * i, n, i):
                    primes[p] = False
                
               
        return sum(primes)
	
# Solution 2
class Solution:
    def countPrimes(self, n: int) -> int:
        notPrimes = [False] * n
        
        count = 0
        for i in range(2, n):
            if (notPrimes[i] == False):
                count += 1 
                
                j = 2
                while i * j < n:
                    notPrimes[i * j] = True
                    j += 1
                
               
        return count
```

#### Greatest Common Divisor
```python
def gcd(a, b):
	return b==0? a: gcd(b, a%b) 
```

对于 a 和 b 的最大公约数 f(a, b)：
- 如果 a 和 b 均为偶数，f(a, b) = 2*f(a/2, b/2)
- 如果 a 是偶数 b 是奇数，f(a, b) = f(a/2, b)
- 如果 b 是偶数 a 是奇数，f(a, b) = f(a, b/2)
- 如果 a 和 b 均为奇数，f(a, b) = f(b, a-b)

乘 2 和除 2 都可以转换为移位操作
```python
def gcd(a, b):
    if (a < b):
        return gcd(b, a)
  
    if (b == 0):
        return a
	
    isAEven, isBEven = isEven(a), isEven(b)
    if isAEven and isBEven:
        return 2 * gcd(a >> 1, b >> 1)
    elif isAEven and not isBEven:
        return gcd(a >> 1, b)
    elif not isAEven and isBEven:
        return gcd(a, b >> 1)
    else:
        return gcd(b, a - b)
```
note x >> y is equivalent to dividing x with 2^y, and x << y is equivalent to multiplying x with 2^y. 

#### Least Common Multiple
```python
def lcm(a, b):
	return a * b //gcd(a, b)
```

#### [172. Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/)
```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        '''
        fives = n/5 + (n/5)/5 + ((n/5)/5)/5 + ...
        '''
        zero_count = 0
        while n > 0:
            n //= 5
            zero_count += n
        return zero_count

#  O(logn)
class Solution:
    def trailingZeroes(self, n: int) -> int:
        zero_count = 0
        current_multiple = 5
        while n >= current_multiple:
            zero_count += n // current_multiple
            current_multiple *= 5
        return zero_count

    
# a better brute force - O(n)
class Solution:
    def trailingZeroes(self, n: int) -> int:
        zero_count = 0
        for i in range(5, n + 1, 5):
            current = i
            while current % 5 == 0:
                zero_count += 1
                current //= 5

        return zero_count

        
# brute force - factorial calc is worse than O(n)
def trailingZeroes(self, n: int) -> int:
        
    # Calculate n!
    n_factorial = 1
    for i in range(2, n + 1):
        n_factorial *= i
    
    # Count how many 0's are on the end.
    zero_count = 0
    while n_factorial % 10 == 0:
        zero_count += 1
        n_factorial //= 10
        
    return zero_count
```

#### [462. Minimum Moves to Equal Array Elements II](https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/)
proof:

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/462.%20Minimum%20Moves%20to%20Equal%20Array%20Elements%20II.png" width="900">

```python
class Solution(object):
    def minMoves2(self, nums):
        n = len (nums)
        mid = sorted (nums) [n // 2]
        res = sum (abs (i - mid) for i in nums)
        return res
```

#### [169. Majority Element](https://leetcode.com/problems/majority-element/description/)
given that it is impossible (in both cases) to discard more majority elements than minority elements, we are safe in discarding the prefix and attempting to recursively solve the majority element problem for the suffix. Eventually, a suffix will be found for which count does not hit 0, and the majority element of that suffix will necessarily be the same as the majority element of the overall array.
```python
# O(n) time and O(1) space
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate
```
```python
# O(n) time and O(n) space
class Solution:
    def majorityElement(self, nums):
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)
```

#### [367. Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/)

<img src="https://github.com/lilywxc/Leetcode/blob/main/pictures/367.%20Valid%20Perfect%20Square.png" width="700">

```python
# Newton's Method
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        x = num // 2
        while x * x > num:
            x = (x + num // x) // 2
            
        return x * x == num
```
```python
# Binary search
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        l, r = 2, num // 2
        
        while l <= r:
            m = l + (r - l) // 2
        
            square = m * m
            if square == num:
                return True
            if square > num:
                r = m - 1
            else:
                l = m + 1
        
        return False
```

#### [628. Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/)
```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        min1, min2 = float('inf'), float('inf')
        max1, max2, max3 = float('-inf'), float('-inf'), float('-inf')
        
        for n in nums:
            if n <= min1:
                min2 = min1
                min1 = n
            elif n <= min2:
                min2 = n
            
            if n >= max1:
                max3 = max2
                max2 = max1
                max1 = n
            elif n >= max2:
                max3 = max2
                max2 = n
            elif n >= max3:
                max3 = n
                
        return max(min1 * min2 * max1, max1 * max2 * max3)
```

## Bit Computation
* [504. Base 7](#504-Base-7)
* [405. Convert a Number to Hexadecimal](#405-Convert-a-Number-to-Hexadecimal)
* [168. Excel Sheet Column Title](#168-Excel-Sheet-Column-Title)
* [67. Add Binary](#67-Add-Binary)
* [415. Add Strings](#415-Add-Strings)
* [461. Hamming Distance](#461-Hamming-Distance)
* [136. Single Number](#136-Single-Number)
* [268. Missing Number](#268-Missing-Number)
* [exchange two integers without extra variables](#exchange-two-integers-without-extra-variables)
* [190. Reverse Bits](#190-Reverse-Bits)
* [231. Power of Two](#231-Power-of-Two)
* [326. Power of Three](#326-Power-of-Three)
* [342. Power of Four](#342-Power-of-Four)
* [693. Binary Number with Alternating Bits](#693-Binary-Number-with-Alternating-Bits)
* [476. Number Complement](#476-Number-Complement)
* [371. Sum of Two Integers](#371-Sum-of-Two-Integers)
* [318. Maximum Product of Word Lengths](#318-Maximum-Product-of-Word-Lengths)
* [338. Counting Bits](#338-Counting-Bits)


**^**: 0^1 = 1, 1^1 = 0, 0^0 = 0
x ^ 0s = x      
x ^ 1s = ~x -> 位级反转     
x ^ x = 0   -> 去重    

**&**: 0&1 = 0, 1&1 = 1, 0&0 = 0 -> 掩码操作
x & 0s = 0      
x & 1s = x      
x & x = x      

**|**：0|1 = 1, 1|1 = 1, 0|0 = 0 -> 设值操作
x | 0s = x
x | 1s = 1s
x | x = x

the number of bits in xx equals to logx + 1

useful tricks:
- x ^ x will remove all duplicates
- x & (x - 1) removes the rightmost bit of '1'
- x & (-x) will keep the rightmost bit of '1' and set all other bits to 0s, where -x = ~x + 1

#### [504. Base 7](https://leetcode.com/problems/base-7/)
7进制
```python
class Solution:
    def convertToBase7(self, num: int) -> str:        
        n, res = abs(num), ''
        
        if num== 0:
            return '0'
        
        while n:
            n, r = divmod(n, 7)
            res = str(r) + res

        return '-' * (num < 0) + res
```

#### [405. Convert a Number to Hexadecimal](https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/)
16进制
```python
class Solution:
    def toHex(self, num: int) -> str:
        if num == 0: 
            return '0'
        
        num = num & 0xFFFFFFFF # same as num = num + 2**32
        dic = '0123456789abcdef'
        res = ''
        
        while num:
            num, r = divmod(num, 16)
            res = str(dic[r]) + res

        return res
```

#### [168. Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/)
26进制
```python
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        if columnNumber == 0:
            return "" 

        q, r = divmod(columnNumber - 1, 26)
        return self.convertToTitle(q) + chr(r + ord('A'))
       
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        res = ''
        while columnNumber:
            columnNumber, r = divmod(columnNumber - 1, 26)
            res = chr(r + ord('A')) + res
            
        return res
```

#### [67. Add Binary](https://leetcode.com/problems/add-binary/description/)
```python
# bit manipulation
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        # ex: a = '11', b = '1'
        x, y = int(a, 2), int(b, 2) # x = 3, y = 1

        while y:
            print('x, y:', x, y)
            answer = x ^ y
            print('x ^ y:', answer)
            carry = (x & y) << 1
            print('x & y << 1:', carry)
            x, y = answer, carry
            
        return bin(x)[2:]

# Bit-by-bit computation      
class Solution:
    def addBinary(self, a, b) -> str:
        n = max(len(a), len(b))
        a, b = a.zfill(n), b.zfill(n)
        
        carry = 0
        answer = []
        for i in range(n - 1, -1, -1):
            if a[i] == '1':
                carry += 1
            if b[i] == '1':
                carry += 1
                
            if carry % 2 == 1:
                answer.append('1')
            else:
                answer.append('0')
            
            carry //= 2
        
        if carry == 1:
            answer.append('1')
        answer.reverse()
        
        return ''.join(answer)
```

#### [415. Add Strings](https://leetcode.com/problems/add-strings/)
```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = []

        carry = 0
        p1 = len(num1) - 1
        p2 = len(num2) - 1
        while p1 >= 0 or p2 >= 0:
            x1 = ord(num1[p1]) - ord('0') if p1 >= 0 else 0
            x2 = ord(num2[p2]) - ord('0') if p2 >= 0 else 0
            
            carry, value = divmod(x1 + x2 + carry, 10)
            res.append(value)
            
            p1 -= 1
            p2 -= 1
        
        if carry:
            res.append(carry)
        
        return ''.join(str(x) for x in res[::-1])
```

#### [461. Hamming Distance](https://leetcode.com/problems/hamming-distance/)
```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
```
```python
class Solution(object):
    def hammingDistance(self, x, y):
        xor = x ^ y
        count = 0
        while xor:
            print(xor, bin(xor)[2:], xor&1)
            if xor & 1:
                count += 1
            xor = xor >> 1
            
        return count
```
x & (x - 1) removes the rightmost bit of '1'
```python
class Solution:
    def hammingDistance(self, x, y):
        xor = x ^ y
        count = 0
        while xor:
            count += 1
            # remove the rightmost bit of '1'
            xor = xor & (xor - 1)
            
        return count
```

#### [136. Single Number](https://leetcode.com/problems/single-number/)
x ^ x = 0, x ^ 0 = x, (a ^ b) ^ c = a ^ (b ^ c) = (a ^ c) ^ b
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        x = 0
        for i in nums:
            x = x ^ i
        return x
```

#### [268. Missing Number](https://leetcode.com/problems/missing-number/)
```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        for i, num in enumerate(nums):
            n = n ^ i ^ num
            
        return n
```

#### [260. Single Number III](https://leetcode.com/problems/single-number-iii/description/)
x & (-x) will keep the rightmost bit of '1' and set all other bits to 0s, where -x = ~x + 1
```python
class Solution:
    def singleNumber(self, nums: int) -> List[int]:
    
        bitmask = 0
        for num in nums:
            bitmask = bitmask ^ num # filter out duplicates
            
        # rightmost 1-bit diff between x and y
        diff = bitmask & (-bitmask)
        
        x = 0
        for num in nums:
            if num & diff:
                x = x ^ num # filter out y and duplicates
        
        y = bitmask ^ x # filter out x
        return [x, y]
```

#### exchange two integers without extra variables
```python
a = a ^ b
b = a ^ b
a = a ^ b
```

#### [190. Reverse Bits](https://leetcode.com/problems/reverse-bits/description/)
```python
class Solution:
    def reverseBits(self, n: int) -> int:
        ret, power = 0, 31
        while n:
            ret += (n & 1) << power
            n = n >> 1
            power -= 1
        return ret
```

#### [231. Power of Two](https://leetcode.com/problems/power-of-two/)
a power of two in binary representation is one 1-bit, followed by some zeros, e.g., 2 = 10, 4 = 100, 8 = 1000. <br />
x & (-x) will keep the rightmost bit of '1' and set all other bits to 0s, where -x = ~x + 1. <br />
since a power of two contains just one 1-bit, this operation will result in the same x, i.e., x & (-x) == x. <br />
Other numbers have more than 1-bit in their binary representation and hence for them x & (-x) would not be equal to x itself.
```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        
        return n & (-n) == n
```
```python
class Solution(object):
    def isPowerOfTwo(self, n):
        if n == 0:
            return False
        return n & (n - 1) == 0
```
```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:  
        return n > 0 and bin(n).count("1") == 1
```


#### [326. Power of Three](https://leetcode.com/problems/power-of-three/description/)
```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and (3**19) % n == 0
```

#### [342. Power of Four](https://leetcode.com/problems/power-of-four/)
Input number is known to be signed 32 bits integer, i.e. x <= 2^31 - 1. Hence the max power of four to be considered is log_4(2^31 - 1)] = 15. Voila, here is all 16 possible answers: 4^0, 4^1, ...4^15
```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        max_power = 15
        nums = [1] * (max_power + 1)
        
        for i in range(1, max_power + 1):
            nums[i] = 4 * nums[i - 1]
        
        return n in nums
```
If num is a power of four x = 4^a, then a = log_4(x) = (1/2)*log_2(x) is an integer. Thus, we simply need to check if log_2(2) is even
```python
from math import log2
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        return num > 0 and log2(num) % 2 == 0
```
In the case of power of four, 1-bit is at even position: bit 0, bit 2, bit 4, etc. e.g., 4 = 100, 8 = 10000 <br />
Hence, power of four would make a zero in a bitwise AND with number (101010...10)_2 = (aaaaaaaa)_16 = 0xaaaaaaaa
```python
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        return num > 0 and num & (num - 1) == 0 and num & 0xaaaaaaaa == 0
```

#### [693. Binary Number with Alternating Bits](https://leetcode.com/problems/binary-number-with-alternating-bits)
```python
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        '''
        n =         1 0 1 0 1 0 1 0
        n >> 1      0 1 0 1 0 1 0 1
        n ^ n>>1    1 1 1 1 1 1 1 1
        n           1 1 1 1 1 1 1 1
        n + 1     1 0 0 0 0 0 0 0 0
        n & (n+1)   0 0 0 0 0 0 0 0
        '''
        tmp = n ^ (n >> 1)
        return tmp & (tmp + 1) == 0
```

#### [476. Number Complement](https://leetcode.com/problems/number-complement/)
detailed picture explanation: [approach 4](https://leetcode.com/problems/number-complement/solution/)
```python
class Solution:
    def findComplement(self, num: int) -> int:
        # construct 1...1 bitmask with same length as num
        bitmask = num
        bitmask |= (bitmask >> 1)
        bitmask |= (bitmask >> 2)
        bitmask |= (bitmask >> 4)
        bitmask |= (bitmask >> 8)
        bitmask |= (bitmask >> 16)

        return bitmask ^ num
    
# a different way to create all-one mask with same length as num
# n = floor(log2(num)) + 1        
# bitmask = (1 << n) - 1
```

#### [371. Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/)
similar question to [67. Add Binary](https://leetcode.com/problems/add-binary/description/)
```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF # bitmask of 32 1-bits
        while b != 0:
            answer = (a ^ b) & mask
            carry = ((a & b) << 1) & mask
            
            a, b = answer, carry
            
        max_int = 0x7FFFFFFF
        return a if a < max_int else ~(a ^ mask)
```

#### [318. Maximum Product of Word Lengths](https://leetcode.com/problems/maximum-product-of-word-lengths/)
```python
from collections import defaultdict
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        hashmap = defaultdict(int)
        bit_number = lambda ch : ord(ch) - ord('a')
        
        for word in words:
            bitmask = 0
            for ch in word:
                bitmask |= 1 << bit_number(ch)
            # there could be different words with the same bitmask, ex. ab and aabb. We store the longest length
            hashmap[bitmask] = max(hashmap[bitmask], len(word))
        
        max_prod = 0
        for word1 in hashmap:
            for word2 in hashmap:
                if word1 & word2 == 0:
                    max_prod = max(max_prod, hashmap[word1] * hashmap[word2])
                    
        return max_prod
```

#### [338. Counting Bits](https://leetcode.com/problems/counting-bits/)
```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        
        for x in range(1, n + 1):
            dp[x] = dp[x & (x - 1)] + 1

        return dp         
```
```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        
        def pop_count(x: int) -> int:
            count = 0
            while x != 0:
                x &= x - 1 # zeroing out the least significant nonzero bit
                count += 1
            return count
            
        ans = [0] * (n + 1)
        for x in range(n + 1):
            ans[x] = pop_count(x)
    
        return ans   
```
