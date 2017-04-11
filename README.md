## SUMMARY
Design and implement a power efficient Parallel Distributed Key Value store on a cluster of raspberry PIs

---

## BACKGROUND

Data is growing at a very fast rate. So, we need a database design which has high scalability, high availability and operational simplicity. Key value database/store provides all these advantages. A key-value store/database is a simple database that uses an associative array as the fundamental data model where each key is associated with one and only one value in a collection. This relationship is referred to as a key-value pair.

Following operation will be supported by our key value store:
Get - Retrieve value for the given key
Put - Store value for the given key
Delete - delete the key-value pair for the given key

As we have discussed, size of database can increase by many folds. So, we need to design a system which can scale our key value store across many machines. Our key-value store will be distributed over cluster of PIs using consistent hashing. The consistent hashing ring will consist of a single master server, and all other servers will be slaves. We are planning to use Raspberry PIs for creating cluster because operations performed on key value store are not compute intensive and thus there is no need of using power hungry andrew machines.

Advantages of distributing load across cluster of PIs:
Performance gain by executing different requests parallely
Power and cost efficient (As we are using cheap and power efficient machines)
Faster data access as data will be distributed across cluster and accessing small database is always efficient and faster

---

## THE CHALLENGE

In most of the cases it is simple to achieve a close to desired performance by just adding more machines to the clusters. But for problems such as key-value storage which are not compute intensive and for systems where cost and/or power is concern it is important to utilize cheap and power efficient machines to achieve the desired goal. We are trying to achieve the same by running our distributed key-value storage on a pi cluster. 
We aim to get our hands dirty with working on pi and to beat the performance observed on andrew machines. 

> ### Workload 
This problem has a high communication to computation ratio as it just needs to fetch the value and return. Further it is important to have an optimal usage of the cache, as any cache miss will increase the fetch time exponentially. So we need to maintain a cache with respect to every storage node in order to minimize the communication. 

> ### Constraints 
Considering the number of workers we plan to have in the system, we need to ensure workload balance because the nature of queries cannot be predicted and we need to have dynamic policies to balance the work.

---

## RESOURCES
### Hardware
Four Raspberry PIs (Type 3 Model B 1.2 GHZ 64-bit quad core ARMv8 CPU)

### Software
Golang
C

We are planning to take reference from a Distributed System project. That project was done with the sole intention of distributing the load and was implemented in GoLang. We are planning to focus more on performance part of it and thus planning to implement it in C language. We would also be planning to take GoLang Project as a starter code to implement the project in GoLang for the described set-up and then compare the performance of both implementations and also compare them with their implementations on andrew machines. 

---

## GOALS AND DELIVERABLES

### Plan to achieve

1\. Successfully install a power efficient Parallel Distributed Key Value store.
We are planning to create a cluster of power efficient Raspberry PIs, this will help us to benchmark the power usage of bulky andrew machines and power efficient PIs. This will also help us to benchmark the throughput between distributed Key Value store and Key Value store on single machine
<br>

2\. Performance Benchmarking
Our goal is to implement this design in both C and Go and compare the difference in performance for both these approaches. Also we want to deploy both these implementations and compare their performance as well. 
<br>

3\. Optimized Solution
We also want thoroughly optimize the solution and come up with a good design. This wood require a thorough creation of a test base that would evaluate the designs in a variety of ways. With this test base we hope to find parts of design that would require optimization and expect to re-implement this optimization elegantly. 
<br>

### Hope to achieve

Implement an advanced load balancer that would dynamically change policies in order to adjust with the variations of incoming requests. 
Implement parallel data compression algorithm to reduce the network traffic by compressing large data values
Persist the in-memory key-value on non-volatile storage to increase durability

---

## PLATFORM CHOICE
One of our main goal is to use a cheap and power efficient machine to obtain high performance. Using Raspberry is perfect for the use case. 

---

## SCHEDULE

- April 11-17: Re-familiarize with GoLang and C RPC package. Order Hardware
<br>
- April 18-24: Setup Pi Cluster and implement Go version of the design
<br>
- April 25-May 1: Implement C version, initial test base and Final exam
<br>
- May 2-9: More advanced test base and performance benchmarking
<br>



