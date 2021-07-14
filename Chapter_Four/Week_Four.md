Machine Learning by Andrew Ng
pdf by CarolusRex

# Neural Networks: Representation

<!-- code_chunk_output -->

- [Non-linear hypotheses](#non-linear-hypotheses)
- [Neurous and the brain](#neurous-and-the-brain)
- [Model and representation 1](#model-and-representation-1)
- [Model and representation 2](#model-and-representation-2)
- [Examples and intuitions 1](#examples-and-intuitions-1)
- [Examples and intuitions 2](#examples-and-intuitions-2)
- [Multi-class classification](#multi-class-classification)

<!-- /code_chunk_output -->

## Non-linear hypotheses
When number of features is large, it's difficult to calculate parameters in a short time.
## Neurous and the brain
## Model and representation 1
$z = \sum_{i = 0} w_ix_i$
$g(z) = \frac{1}{1 + e^{-z}}$
即$h_\Theta(x) = \frac{1}{1 + e^{-\Theta^Tx}}$

$a_i^{(j)} = g(\sum_{k = 0} \Theta_{ik}^{(j - 1)} x_k)$
$a^{(j)} = g(\Theta^{j - 1}x)$
$\Theta^{j - 1}$ is $s_j * (s_{j - 1} + 1)$
## Model and representation 2
$z^{(2)} = \Theta^{(1)} * a^{(1)}$
$a^{(2)} = g(z^{(2)})$
Add $a_0^{(2)} = 1$.
$z^{(3)} = \Theta^{(2)} * a^{(2)}$
$h_\Theta(x) = a^{(3)} = g(z^{(3)})$
## Examples and intuitions 1
$y = x_1\ XOR\ x_2$
$y = x_1\ XNOR\ x_2$
$y = NOT(x_1\ XOR\ x_2)$
## Examples and intuitions 2
$y = x_1\ AND\ x_2$
$y = x_1\ OR\ x_2$
## Multi-class classification
$h_\Theta(x) \in \mathbb{R}^n$
Training set: $(x^{(i)}, y^{(i)})$
$y_j^{(i)} = 1,\ y_{k \neq j}^{(i)} = 0$
