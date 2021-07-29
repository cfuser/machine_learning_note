Machine Learning by Andrew Ng
pdf by CarolusRex

# Advice for applying machine learning

<!-- code_chunk_output -->

- [Deciding what to try next](#deciding-what-to-try-next)
- [Evaluating a hypothesis](#evaluating-a-hypothesis)
- [Model selection and training/validation/test sets](#model-selection-and-trainingvalidationtest-sets)
- [Diagnosing bias vs. variance](#diagnosing-bias-vs-variance)
  - [Regularization and bias/variance](#regularization-and-biasvariance)
- [Learning Curves](#learning-curves)
- [Deciding what to try next (revisited)](#deciding-what-to-try-next-revisited)
- [Neural networks and overfitting](#neural-networks-and-overfitting)

<!-- /code_chunk_output -->


## Deciding what to try next
hypothesis makes unacceptably large errors in its predictions.
Some method to improve:
1. Get more training examples;
2. Try smaller sets of features;
3. Try getting additional features;
4. Try adding polynomial features;
5. Try decreasing $\lambda$;
6. Try increasing $\lambda$.

## Evaluating a hypothesis
devide datas into training data and test data;
some parameter to evaluate the hypothesis: $J_{test}$ or $misclassification\      error$

## Model selection and training/validation/test sets
fit the parameter by training data, compute the error by validation data and choose the best parameter, get the final error by test data.

## Diagnosing bias vs. variance
|				|						|												|
|-|-|-|
|Underfit	|High bias		|$J_{train}(\theta)$ will be high and $J_{cv}(\theta) \approx J_{train}(\theta)$	|
|Overfit	|High variance	|$J_{train}(\theta)$ wiil be low and $J_{cv}(\theta) >> J_{train}(\theta)$				|
![erro![在这里插入图片描述](https://img-blog.csdnimg.cn/99b4fcd7a8664293bbac99bf94433e42.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
r-d\](https://img-blog.csdnimg.cn/119e1f8fffaa485da897e9e0c9cc48dd.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)](https://img-blog.csdnimg.cn/44a19f0355c64dff8fb6b6c7fba8f832.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)

### Regularization and bias/variance
|||
|-|-|
|Large $\lambda$|High bias(Underfit)	|
|Small $\lambda$|High variance(Overfit)|
![J-lambda](https://img-blog.csdnimg.cn/189b5d8f9e774cad8d8a2630f6389d47.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)

## Learning Curves
![error-m](https://img-blog.csdnimg.cn/7f665e062ee54e86b3aa05ed68205aff.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
![error-m of high bias](https://img-blog.csdnimg.cn/18d9bc293d4b4b16bc1b87c35e4766f0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

![error-m of high variance](https://img-blog.csdnimg.cn/6c3b8673ee2443889d444c06d7779afa.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nhcm9sdXNSZXg=,size_16,color_FFFFFF,t_70)
If a learning algorithm is suffering from high variance, getting more data is likely to help.

## Deciding what to try next (revisited)
1. Get more training examples $\rightarrow$ fixed high variance;
2. Try smaller sets of features $\rightarrow$ fixed high variance;
3. Try getting additional features $\rightarrow$ fixed high bias;
4. Try adding polynomial features $\rightarrow$ fixed high bias;
5. Try decreasing $\lambda$ $\rightarrow$ fixed high bias;
6. Try increasing $\lambda$ $\rightarrow$ fixed high vriance.

## Neural networks and overfitting
|size of Neural Nerwork |size of parameters         																							|complexity						|
|-|-|-|
|Small Neural Netwrok	|fewer parameters; more prone to underfitting;  |Computationally Cheaper		|
|Large Neural Network	|more paramaters; more prone to overfitting     |Computationally more expensive	|