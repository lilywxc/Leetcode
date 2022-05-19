# Leetcode - Data Structure
* [LinkedList](#LinkedList)
   * [160. Intersection of Two Linked Lists](#160-Intersection-of-Two-Linked-Lists)
   * [206. Reverse Linked List](#206-Reverse-Linked-List)
   * [21. Merge Two Sorted Lists](#21-Merge-Two-Sorted-Lists)
   * [83. Remove Duplicates from Sorted List](#83-Remove-Duplicates-from-Sorted-List)
   * [19. Remove Nth Node From End of List](#19-Remove-Nth-Node-From-End-of-List)
   * [24. Swap Nodes in Pairs](#24-Swap-Nodes-in-Pairs)
   * [445. Add Two Numbers II](#445-Add-Two-Numbers-II)
   * [234. Palindrome Linked List](#234-Palindrome-Linked-List)
   * [725. Split Linked List in Parts](#725-Split-Linked-List-in-Parts)
   * [328. Odd Even Linked List](#328-Odd-Even-Linked-List)
* [Tree](#Tree)
   * [104. Maximum Depth of Binary Tree](#104-Maximum-Depth-of-Binary-Tree)
   * [110. Balanced Binary Tree](#110-Balanced-Binary-Tree)
   * [543. Diameter of Binary Tree](#543-Diameter-of-Binary-Tree)
   * [226. Invert Binary Tree](#226-Invert-Binary-Tree)
   * [617. Merge Two Binary Trees](#617-Merge-Two-Binary-Trees)
   * [112. Path Sum](#112-Path-Sum)
   * [437. Path Sum III](#437-Path-Sum-III)
   * [572. Subtree of Another Tree](#572-Subtree-of-Another-Tree)

### LinkedList
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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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

#### [725. Split Linked List in Parts](725. Split Linked List in Parts)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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

#### Tree

#### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

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
        num_node_level = 1
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

#### [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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
 ```python
 
 ```
 
 #### [572. Subtree of Another Tree]()

