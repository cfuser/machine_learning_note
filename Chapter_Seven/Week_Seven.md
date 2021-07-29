Machine Learning of Andrew Ng
pdf by CarolusRex

# Suppore Vector Machines

<!-- code_chunk_output -->

- [Optimization objective](#optimization-objective)
- [Large Margin Intuition](#large-margin-intuition)
- [SVM derivation](#svm-derivation)
- [Multi-class classification](#multi-class-classification)

<!-- /code_chunk_output -->

## Optimization objective
alternative view of logistic regression
$h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}$
![h_theta(x)](https://img-blog.csdnimg.cn/858beb66f40a4f80a836525eb3737a3f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
 If $y == 1$, we want $h_\theta(x) \approx 1,\ \theta^Tx >> 0$;
 If $y == 0$, we want $h_\theta(x) \approx 0,\ \theta^Tx << 0$.
And we use linear to replace the cost function, like fellow.
![y == 1](https://img-blog.csdnimg.cn/be8eb5c923ee42fd81a05bab94fb08ea.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
![y == 0](https://img-blog.csdnimg.cn/65edaba4c45c487688bf1872ba8771b1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)

## Large Margin Intuition
![在这里插入图片描述](https://img-blog.csdnimg.cn/7049ab4496234f3a94ce0313f7d2db58.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
Here the cost function $J$ has the regularization.
$C = \frac{1}{\lambda}$
When $C$ is large, we want $C \cdot 0$ and $min\ \frac{1}{2} \sum \theta_j^2$, which is prone to overfitting.
If C is large, then we get higher variance/lower bias
If C is small, then we get lower variance/higher bias

The other parameter we must choose is $\sigma^2$   from the Gaussian Kernel function:

With a large $\sigma^2$, the features fi vary more smoothly, causing higher bias and lower variance.

With a small $\sigma^2$, the features fi vary less smoothly, causing lower bias and higher variance.
## SVM derivation
[SVM-1——derivation of target and convex optimization](https://blog.csdn.net/CarolusRex/article/details/119143710)(blog) or [SVM-1](https://github.com/cfuser/machine_learning_note/tree/master/Chapter_Seven/SVM-1.md)(github)
[SVM-2——nonlinear, kernel and SMO derivation](https://editor.csdn.net/md/?articleId=119150403)(blog) or [SVM-2](https://github.com/cfuser/machine_learning_note/tree/master/Chapter_Seven/SVM-2.md)(github)

Mercer's Theorem: 任何半正定矩阵都能作为核函数。

## Multi-class classification
one-vs-all method, pick class $i$ with the largest $(\Theta^{(i)})^Tx$.

If n is large (relative to m), then use logistic regression, or SVM without a kernel (the "linear kernel")

If n is small and m is intermediate, then use SVM with a Gaussian Kernel

If n is small and m is large, then manually create/add more features, then use logistic regression or SVM without a kernel.

In the first case, we don't have enough examples to need a complicated polynomial hypothesis. In the second example, we have enough examples that we may need a complex non-linear hypothesis. In the last case, we want to increase our features so that logistic regression becomes applicable.

Note: a neural network is likely to work well for any of these situations, but may be slower to train.