# Leetcode - Data Structure
* [LinkedList](#LinkedList)
    * [160. Intersection of Two Linked Lists](#160-Intersection-of-Two-Linked-Lists)


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
