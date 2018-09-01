# Cuckoo Cycle

## Introduction

The proposed Cockoo Cycle PoW is based on finding certain subgraphs in large pseudo-random graphs.  

### Concept

For a graph $G(V,E)$, we choose edges from the output of a keyed hash function $h$, whose keys could be chosen uniformly at random.

$N$: node number, $ N \leq 2^{W_o} $
$M$: edge number, $ M \leq 2^{W_i-1} $
$k$: key, $k \in \{0,1\}^K$

Fix a keyed hash function $h:\{0,1\}^K \times \{0,1\}^{W_i} \rightarrow \{0,1\}^{W_o}$, and a small graph $H$ as a target subgraph.

Each $k \in \{0,1\}^K$ generates a graph $G_k = (V,E)$ where $V=\{v_0,...,v_{N-1}\}$ is the node set, and
$E = \{(v_{h(k,2i)\mod N}, v_{h(k,2i+1)\mod N}\} | i\in[0,...,M-1]\}$. Each vector in $\{0,1\}^N$ is also an N-bit binary number.

A simple variation generates random bipartite graphs:
$G_k(V_0 \cup V_1,E)$ assuming that N is even, $V_0=\{v_0,v_2,...,v_{N-2}\}$, $V_1=\{v_1,v_3,...,v_{N-1}\}$ and
$E=\{ (v_{2(h(k,2i)\mod \frac{N}{2})}, v_{2(h(k,2i+1)\mod\frac{N}{2})+1})|i\in[0,...,M-1] \}$
Nodes inside $V_0$ and $V_1$ are not adjacent to other nodes in the same set, thus $G_k(V_0 \cup V_1, E)$ is a bipartite graph.

## Parameter Specification

In Cuckoo Cycle, the target subgraph $H$ is an $L$-cycle. The hash function $h$ is siphash with a $K = 128$ bit key, $W_i=W_o=64$ input and output bits, $N\leq2^{64}$ a 2-power, M = N/2. Inserting items in a Cuckoo hashtable naturally leads to cycle formation in random bipartite graphs, that is, *Cockoo graphs*.

## Cuckoo graph

### Definition

A Cuckoo hashtable consists of two same-sized tables each maps a key to a table location, providing two locations for each key. When inserting a new key, if both locations are already occupied by keys, then one of them is kicked out and inserted in its alternate location, possibly displacing yet another key. The process is repeated until either an empty location is found, or a maximum number of iterations is reached. The latter is bound to happen if cycles have formed in the *Cuckoo graph*.

A Cuckoo graph is a bipartite graph with a node for each possible location and an edge for every inserted key, connecting the two hashed locations generated with a Cuckoo hashtable.  

### Cycle Detection

We enumerate the $M$ nonces to generate $M$ edges in a *directed* Cuckoo graph. The edge for a key is directed from the location where it resides to its alternate location. Moving a key to its alternate location thus corresponds to reversing its edge. Therefore the outdegree of every node in this graph is either 0 or 1, empty or occupied. When there are no cycles yet, the graph is a forest, a disjoint union of trees. In each tree, all edges are directed to its root, the only node in the tree with outdegree 0.

Initially there are $N$ singleton trees consisting of  individual nodes which are all roots. Inserting a new key causes a cycle if and only if its two endpoints are nodes in the same tree, which we can test by following the path from each endpoint to its root. If the two endpoints belong to different trees, we have two paths each starts from one endpoint to its root. We reverse all edges of the shorter one, and create the edge for the new key, there by joining the two trees into one.

