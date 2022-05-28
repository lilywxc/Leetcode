# Leetcode - Data Structure
* [LinkedList](#LinkedList)
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
* [Tree](#Tree)
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
* [Stack and Queue](#Stack-and-Queue) 
    * [232. Implement Queue using Stacks](#232-Implement-Queue-using-Stacks)
    * [225. Implement Stack using Queues](#225-Implement-Stack-using-Queues)
    * [155. Min Stack](#155-Min-Stack)
    * [20. Valid Parentheses](#20-Valid-Parentheses)
    * [496. Next Greater Element I](#496-Next-Greater-Element-I)
    * [739. Daily Temperatures](#739-Daily-Temperatures)
    * [503. Next Greater Element II](#503-Next-Greater-Element-II)
* [HashMap](#HashMap)
    * [1. Two Sum](#1-Two-Sum)
    * [217. Contains Duplicate](#217-Contains-Duplicate)
    * [594. Longest Harmonious Subsequence](#594-Longest-Harmonious-Subsequence)
    * [128. Longest Consecutive Sequence](#128-Longest-Consecutive-Sequence)
* [String](#String)
    * [242. Valid Anagram](#242-Valid-Anagram)
    * [409. Longest Palindrome](#409-Longest-Palindrome)
    * [205. Isomorphic Strings](#205-Isomorphic-Strings)
    * [647. Palindromic Substrings](#647-Palindromic-Substrings)
    * [5. Longest Palindromic Substring](#5-Longest-Palindromic-Substring)
    * [9. Palindrome Number](#9-Palindrome-Number)
    * [696. Count Binary Substrings](#696-Count-Binary-Substrings)
* [Array and Matrix](#Array-and-Matrix)
    * [283. Move Zeroes](#283-Move-Zeroes)
    * [566. Reshape the Matrix](#566-Reshape-the-Matrix)
    * [240. Search a 2D Matrix II](#240-Search-a-2D-Matrix-II)
    * [378. Kth Smallest Element in a Sorted Matrix](#378-Kth-Smallest-Element-in-a-Sorted-Matrix)

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

#### [725. Split Linked List in Parts](https://leetcode.com/problems/split-linked-list-in-parts/)
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

#### BFS
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

#### DFS

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

#### Recursion
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

#### BST
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
``python        
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

#### Trie

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

#### Stack and Queue

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

#### HashMap

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

#### String

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

#### Array and Matrix

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
                nums[insertPos] = num    
                insertPos += 1

        while insertPos < len(nums):
            nums[insertPos] = 0
            insertPos += 1
```
    
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
# # # # # ? . .
# # # ? . . . .
# ? . . . . . .   "#" means pair already in the output
# ? . . . . . .   "?" means pair currently in the queue
# ? . . . . . .
? . . . . . . .
. . . . . . . .
```
```python
# let X = min(K, N), it's O(X + KlogX), 
# O(X): heap construction
# O(KlogX): K iterations of popping and pushing from a heap of X elements
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        N = len(matrix)
        
        minHeap = []
        for r in range(min(k, N)):
            minHeap.append((matrix[r][0], r, 0))
        
        heapq.heapify(minHeap)    
        
        while k:
            
            element, r, c = heapq.heappop(minHeap)
            
            if c < N - 1:
                heapq.heappush(minHeap, (matrix[r][c + 1], r, c + 1))
            
            k -= 1
        
        return element  
```
