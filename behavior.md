### 1. Receive the Question
Coding questions tend to be vague and underspecified on purpose to gauge the candidate's attention to detail and carefulness. <br />
Ask at least 2-3 clarifying questions.
1. Paraphrase and repeat the question back at the interviewer to ensure you understood the question
2. Clarify assumption <br />
    - Clarify if the given diagram is a tree or a graph. If a graph allows for cycles, then a naive recursive solution would not work
    - Can you modify the original array / graph / data structure in any way?
    - How is the input stored?
    - Is the input array sorted? (e.g. for deciding between binary / linear search)
2. Clarify input value range
3. Clarify input value format
    - Values: Negative? Floating points? Empty? Null? Duplicates? Extremely large?
4. Work through a simplified example to ensure you understood the question

### 2. Discuss approachs
Do **not** jump into coding yet.<br />
This is a **2-way** discussion on approaches to take for the question, including analysis of the time and space complexity.<br />
1. Explain a few approaches that you could take at a high level (don't go too much into implementation details), and discuss time/space tradeoffs of each approach with your interviewer as if the interviewer was your coworker and you all are collaborating on a problem.
2. State and explain your proposed approach(es) and provide the big O time and space complexity
3. Agree on the most ideal approach and optimize it. Identify repeated/duplicated/overlapping computations and reduce them via caching. 

### 3. Code out your solution while talking through it
Do **not** start coding until the interviewer gives you the green light<br />
Explain what you are trying to achieve as you are coding / writing. Compare different coding approaches where relevant.
