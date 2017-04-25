## SUMMARY
We plan to implement an efficient algorithm to count the number of triangles
parallelly on a GPU. Also compare the performance of our implementation with the
reference implementation and previous 15-418 implementation of the students. 

---

## BACKGROUND

Graphs can be used to model interactions between entities in a broad spectrum of
applications. Graphs can represent relationships in social media, the World Wide
Web, biological and genetic interactions, co-author networks, citations, etc.
Therefore, understanding the underlying structure of these graphs is becoming
increasingly important, and one of the key techniques for understanding is based
on finding small subgraph patterns. The most important such subgraph is the
triangle.

Many important measures of a graph are triangle-based, such as the clustering
coefficient and the transitivity ratio. The clustering coefficient is frequently
used in measuring the tendency of nodes to cluster together as well as how much
a graph resembles a small-world network. The transitivity ratio is the
probability of wedges (three connected nodes) forming a triangle.

So, our code with take a graph G as input and output a single integer which
would represent the number of distinct triangles present in the graph.
Previously, many sequential algorithms have been implemented to solve this
problem. We are planning to implement a parallel algorithm which will run on
GPUs to solve this problem.

---

## THE CHALLENGE

t has never been simple to solve the problem of counting the number of
triangles, moreover not so exciting doing it in O(n^3) and O(3xn) space. There
are many implementations available using techniques such as map-reduce,
GraphLab, etc. 

Our reference paper claims to have a very efficient implementation, using
subgraph matching to a triangle pattern, in recent times and we will try to have
our own implementation of their algorithm and beat their performance as well as
the similar previous years 15-418 projects.

As mentioned, our biggest challenge would be to optimize the authors algorithm
further and also optimize the memory usage, the most tricky part would be the
communication part between the cores computing the subgraphs.

---

## PLATFORM & RESOURCES

- As per the current plan we will need the basic ghc machines to test and analyze our implementation.
<br>
- To collect candidate edges in our algorithm, we are planning to use “select primitive” from Merrill’s CUB library.
<br>

---

## GOALS AND DELIVERABLES

### Plan to achieve

- We plan to successfully implement the algorithm mentioned in the paper and
achieve at least the same performance. 
<br>
- Beat the previous years project performance.
<br>
- Have a detailed analysis of performance between our implementation and baseline
sequential & parallel algorithms.
<br>

### Hope to achieve

- We want to optimize the given algorithm and come up with a better design. This
  would require a thorough testing framework which would evaluate the designs in
  a variety of ways. With this test framework we hope to find parts of design
  that would require optimization and expect to re-implement this optimization
  elegantly. 

---

## SCHEDULE

- April 11-17: Thoroughly go through the reference paper and understand the algorithm. 
<br>
- April 18-24: Come up with a draft design of the implementation and gather libraries and test cases for benchmarking.  
<br>
- April 25-May 1: Implement the first version and prepare for final exam 
<br>
- May 2-9: Optimize the initial implementation and perform analysis. 
<br>


---
---
---

## CHECKPOINT REPORT

### As per schedule

- April 11-13: Revisited our original proposal and after discussion with Prof.
  Kayvon, we pivoted from our original idea. After lot of brainstorming, we came
  up with the new project proposal “Parallel Triangle Counting on GPU”
<br>
- April 14-18: Thoroughly went through the reference paper and understood the
  algorithm and its intricacies.
<br>
- April 19-24: Came up with a draft design of the implementation and gathered
  libraries and test cases for benchmarking
<br>

### Pending

- April 25-May 1: Implement the first version and prepare for final exam
<br>
- May 2-9: Optimize the initial implementation and perform analysis
<br>

---

### Progress So far
- After changing our initial project proposal, things went as planned. Till now
  we have gathered all the benchmarking test cases (graphs). That is:

  -------------------------------------------------------------------
  <br>
  | Dataset Names       |   #nodes    |   #edges    |   #triangles  |
  <br>
  -------------------------------------------------------------------
  <br>
  | cit-Patents         |  3,774,7683 |   3,037,896 |   7,515,023   |
  <br>
  -------------------------------------------------------------------
  <br>
  | coAuthorsCiteseer   |   227,320   |   1,628,268 |   8,100,000   |
  <br>
  -------------------------------------------------------------------
  <br>
  | coPapersDBLP        |   540,486   |  30,491,458 | 1,300,000,000 |
  <br>
  -------------------------------------------------------------------
  <br>
  | road central        | 14,081,816  |  33,866,826 |    687,000    |
  <br>
  -------------------------------------------------------------------
  <br>
  | soc-LiveJournal1    |   4,847,571 | 137,987,546 |   285,730,264 |
  <br>
  -------------------------------------------------------------------
  <br>
  | com-Orkut           |   3,072,441 | 234,370,166 |   627,584,181 |
  <br>
  -------------------------------------------------------------------
  <br>


- Further, we have spent a lot of time on understanding the k-truss approach of
  the subgraph technique mentioned in the paper. We have made a rough design on
  how to go ahead with the implementation, but we have not spent much time on
  the design part as from the assignments we understand that it is better to
  start with a simple straightforward implementation initially. 

- We are on track with respect to the goals and deliverables stated in our
  proposal. We still believe that we will be able to achieve all the things that
  we have mentioned in our goals. 

- We are planning to show performance analysis of our algorithm using graphs.
  Graphs will basically analyse performance of our algorithm wrt baseline
  implementations.
