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
