Machine Learning by Andrew Ng
pdf by CarolusRex
# Linear Regression with multiple variables

<!-- code_chunk_output -->

- [Mutilple Features](#mutilple-features)
  - [Twp Practice](#twp-practice)
- [Method](#method)
    - [Normal equation](#normal-equation)

<!-- /code_chunk_output -->

## Mutilple Features
**Notation:**
$n$ = number of features
$x^{(i)}$ = input(features) of $i^{th}$ training example.
$x^{(i)}_j$ = value of feature $j$ in $i^{th}$ training example.

**Hypothesis:**
$h_{\theta}(x) = \sum_{i = 0}^{n} \theta_i x_i = \theta^{T}x$
CostFunction $J = \frac{1}{2m}\sum _{i = 1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$
$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) =  \theta_j - \alpha \frac{1}{m}\sum _{i = 1}^{m} [(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}]$

### Twp Practice
1. Feature Scaling
2. Learning Rate

Further, Feature Scaling can be devided into Feature Scaling & Mean Normalization.
**Feature Scaling:** 
Idea: Make sure features are on a similar scale.
Get every feature into approximately $-1 \leqslant x_i \leqslant 1$ range.

**Mean Normalization:**
Replace $x_i$ with $x_i - \mu_i$ to make features have approximately zero mean.(Do not apply to $x_0 = 1$).

Learning Rate:
If $\alpha$ is too small: slow convergence.
If $\alpha$ is too large: $J(\theta)$ may not decrease on every iteration; may not converge.

## Method
Here are two methods to fitting the targer:
1. Gradient descent
2. Normal equation

#### Normal equation
$\frac{\partial}{\partial \theta_j} J(\theta) = 0$ for every $j$
Solve for $\theta_i$
which means $0 = \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m}\sum _{i = 1}^{m} [(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}]$
$X^{T}(X\theta-Y) = 0 \Rightarrow \theta = (X^TX)^{-1}X^TY$, while it's often substitute $pinv(X^TX)$ for $(X^TX)^{-1}$.

|Mehtod |Character	|
|--|--|
|Gradient Descent	|Need to choose $\alpha$<br> Need many Operations.<br> Work well even when $n$ is large.|
|Normal Equation	|No need to choose $\alpha$.<br> Don't need to iterate.<br> Need to compute $(X^TX)^{-1}$, whose time complexity is $O(n^3)$<br> Slow if $n$ is very large.|
