### 1. Receive the Question
Coding questions tend to be vague and underspecified on purpose to gauge the candidate's attention to detail and carefulness. <br />
Ask at least 2-3 clarifying questions.
1. Paraphrase and repeat the question back at the interviewer to ensure you understood the question.
2. Clarify assumption. <br />
    - Clarify if the given diagram is a tree or a graph. If a graph allows for cycles, then a naive recursive solution would not work.
    - Can you modify the original array / graph / data structure in any way?
    - How is the input stored?
    - Is the input array sorted? (e.g. for deciding between binary / linear search)
2. Clarify input value range.
3. Clarify input value format.
    - Values: Negative? Floating points? Empty? Null? Duplicates? Extremely large?
4. Work through a simplified example to ensure you understood the question.

### 2. Discuss approachs
Do **not** jump into coding yet.<br />
This is a **2-way** discussion on approaches to take for the question, including analysis of the time and space complexity.<br />
1. Explain a few approaches that you could take at a high level (don't go too much into implementation details), and discuss time/space tradeoffs of each approach with your interviewer as if the interviewer was your coworker and you all are collaborating on a problem.
2. State and explain your proposed approach(es) and provide the big O time and space complexity.
3. Agree on the most ideal approach and optimize it. Identify repeated/duplicated/overlapping computations and reduce them via caching. 

### 3. Code out your solution while talking through it
Do **not** start coding until the interviewer gives you the green light<br />
1. Explain what you are trying to achieve as you are coding / writing. Compare different coding approaches where relevant. (*e.g. choice of array vs dictionary. In so doing, demonstrate mastery of your chosen programming language.*)
2. Code at a reasonable speed so you can talk through it - but not too slow.
3. Use variable names that explain your code.
4. Ask for permission to use trivial functions without having to implement them. (*e.g. sort(), Counter()*)
5. Write in a modular fashion, going from higher-level functions and breaking them down into smaller helper functions.
6. If you are cutting corners in your code, state that out loud to your interviewer and say what you would do in a non-interview setting. (*e.g. "Under non-interview settings, I would write a regex to parse this string rather than using split() which may not cover certain edge cases."*)
7. use comments if necessary, but not too many as your code should be self-explanatory.
    
### 4. After coding, check your code and add test cases
Once you are done coding, do **not** announce that you are done. <br />
Interviewers expect you to start scanning for mistakes and adding test cases to improve on your code.
1. Read through your code for mistakes, as if it's your first time seeing a piece of code written by someone else, and talk through your process of finding mistakes.
2. Step through your code with a few test cases. 
3. Brainstorm edge cases with the interviewer and add additional test cases.
4. Look out for places where you can refactor.
5. Reiterate the time and space complexity of your code.(*This allows you to remind yourself to spot issues within your code that could deviate from the original time and space complexity*) 
6. Explain trade-offs and how the code / approach can be improved if given more time.


Side notes: 
- before you are given the question, give interviewer a 2 minute self-introduction.
- after solving the questions, ask good final questions ([examples](https://www.techinterviewhandbook.org/final-questions/)) that are tailored to the company and thanks the interviewer.
- send a thank you email later


[Resume](https://www.techinterviewhandbook.org/resume/)
