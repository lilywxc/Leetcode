# Leetcode - Data Structure
* [LinkedList](#LinkedList)
    * [160. Intersection of Two Linked Lists](#160-Intersection-of-Two-Linked-Lists)
    * [206. Reverse Linked List](#206-Reverse-Linked-List)


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
```pyhon
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# iterative
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
            
        return prev
```
```python
# recursive 1
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
# recursive 2
class Solution:
    def reverseList(self, curr: Optional[ListNode], prev = None) -> Optional[ListNode]:
        if not curr:
            return prev
        
        new_head = curr.next
        curr.next = prev
        
        return self.reverseList(new_head, curr)
```
