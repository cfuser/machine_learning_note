Machine Learning by Andrew Ng
pdf by CarolusRex
# Introduction & Supervised Learning

<!-- code_chunk_output -->

- [Defination:](#defination)
- [Machine learning algorithms:](#machine-learning-algorithms)
  - [Supervised Learning:](#supervised-learning)
    - [Gradient descent](#gradient-descent)

<!-- /code_chunk_output -->

## Defination:
“A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.”

## Machine learning algorithms:
1. **Supervised Learning;**
2. **Unsupervised Learning;**
3. others, including Reinforcement learning, recommender systems.

|Method	|right answer	|output	|
|--|--|--|
|	**Supervised Learning** 		|	"right answer" **given**		|**Regression:** Predict continuous valued output|
|	**Unsupervised Learning:** |	"right answer" **ungiven**	|**Classification:** Discrete valued output	|

### Supervised Learning:
**Hyposthesis:** $h_{\theta}(x) = \sum_{i = 0}^{n} \theta_ix^{k}$
**CostFunctionJ:** $J = \frac{1}{2m}\sum _{i = 1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$
**Targer:** choose $\theta$ to minimize CostFunction $J$

#### Gradient descent
$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$
when CostFunctionJ is shown above, $\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i = 1}^{m}[(h_{\theta}(x^{(i)}) - y^{(i)})\cdot \frac{\partial h_{\theta}(x^{(i)})}{\partial \theta_j}]$

**“Batch” Gradient Descent**
**Batch:** Each step of gradient descent  uses all the training examples.