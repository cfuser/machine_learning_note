Machine Learning of Andrew Ng
pdf by CarolusRex

# Large scale machine learning

<!-- code_chunk_output -->

- [Stochastic gradient descent](#stochastic-gradient-descent)
  - [Stochastic gradient descent convergence](#stochastic-gradient-descent-convergence)
- [Online Learning](#online-learning)
- [Map reduce and data parallelism](#map-reduce-and-data-parallelism)

<!-- /code_chunk_output -->

## Stochastic gradient descent
1. Randomly shuffle(reorder) training examples;
2. Repeat { //1 - 10
&emsp;for i = 1, ..., m{
&emsp;&emsp;$\theta_j = \theta_j - \alpha (h_\theta(x^{(i)})- y^{(i)}) \cdot x_j^{(i)}$
&emsp;}
}

Batch gradient descent: Use all $m$ examples in each iteration;
Stochastic gradient descent: Use $1$ example in each iteration;
Mini-batch gradient descent: Use $b$ examples in each iteration.

### Stochastic gradient descent convergence
Learning rate $\alpha$ is typically held constant. Can slowly decrease $\alpha$ over time if we want $\theta$ to converge. (E.g. $\alpha = \frac{const_1}{iterationNumber + const_2}$)

## Online Learning
## Map reduce and data parallelism
