# GraphRox Research Plan

## Checking Viability of approximated graphs

1. Generate a set $S$ of 500 graphs each with 500 nodes and a set $L$ of 500 graphs each with
5,000 nodes. Use edge count $m_i$ to generate scale-free graphs in a Barabasi Albert model.
2. Run various graph embedding algorithms on $S$ and $L$ and their approximated counterparts.
Use $t_j$ as a threshold for the approximation and $b_k$ as a block dimension.
3. For each pair of graph embeddings (a pair consists of the standard embedding and the
approximate embedding and the execution time of each), calculate the error of the approximate
embedding.
4. Repeat steps 2 and 3 with 5 different $t_j$ thresholds and collect data.
4. Repeat steps 2 through 4 with 5 different block dimensions.
5. Repeat steps 1 through 4 with 5 different $m$s.

## Potential Paper titles
* Graph approximation for faster calculation of scale-free graph embeddings
