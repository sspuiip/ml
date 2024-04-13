# 高斯模型

&emsp;&emsp;高斯模型主要用于连续型数据的建模与表示。

## 高斯分布


&emsp;&emsp;（一）**定义**

&emsp;&emsp;高斯分布(Gaussian Distribution)。随机变量$\forall \pmb{x}\in\mathbb{R}^D$服从高斯分布，记为$\pmb{x}\sim\mathcal{N}(\pmb{\mu},\pmb{\Sigma})$的概率密度函数为，

$$
p(\pmb{x}|\pmb{\mu},\pmb{\Sigma})=\frac{1}{(2\pi)^{D/2}|\pmb{\Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}\underbrace{(\pmb{x}-\pmb{\mu})^\top\pmb{\Sigma}^{-1} (\pmb{x}-\pmb{\mu})}_{马氏距离}\right\}
$$(multi-guassian-dist-def)

&emsp;&emsp;上式中$\pmb{\Sigma}^{-1}$为对称可逆矩阵。对方差矩阵$\pmb{\Sigma}$进行特征值分解可知，$\pmb{\Sigma}=\pmb{U}\pmb{\Lambda}\pmb{U}^\top, s.t. \pmb{U}^\top\pmb{U}=\pmb{I}$。从而有$\pmb{\Sigma}^{-1}=\pmb{U}\pmb{\Lambda}^{-1}\pmb{U}^\top$，继续展开可得$\pmb{\Sigma}^{-1}=\sum_{i=1}^d\lambda_i^{-1}\pmb{u}_i\pmb{u}_i^\top$。将$\pmb{\Sigma}^{-1}$代回马氏距离，于是有，

$$
\begin{split}
(\pmb{x}-\pmb{\mu})^\top\pmb{\Sigma}^{-1} (\pmb{x}-\pmb{\mu})&=(\pmb{x}-\pmb{\mu})^\top\sum_{i=1}^d\lambda_i^{-1}\pmb{u}_i\pmb{u}_i^\top(\pmb{x}-\pmb{\mu})\\
&=\sum_{i=1}^d\lambda_i^{-1}(\pmb{x}-\pmb{\mu})^\top\pmb{u}_i\underbrace{\pmb{u}_i^\top(\pmb{x}-\pmb{\mu})}_{y_i}\\
&=\sum_{i=1}^d\frac{y_i}{\lambda_i}
\end{split}
$$(guassian-ellipse)

在二维平面，椭圆方程为$\frac{y_1^2}{\lambda_1}+\frac{y_2^2}{\lambda_2}=1$，由此可知高斯密度函数的等高线和椭圆相似。

&emsp;&emsp;（二）**参数的最大似然估计**

&emsp;&emsp;假设有$N$个独立同分布的随机样本$\pmb{x}_i\sim\mathcal{N}(\pmb{\mu},\pmb{\Sigma})$，则参数$\pmb{\mu},\pmb{\Sigma}$的最大似然估计分别为，

$$
\hat{\pmb{\mu}}=\sum_{i=1}^{N}\pmb{x}_i=\bar{\pmb{x}},\quad \hat{\pmb{\Sigma}}=\frac1N\sum_{i=1}^N(\pmb{x}_i-\bar{\pmb{x}})(\pmb{x}_i-\bar{\pmb{x}})^\top=\frac1N\sum_{i=1}^N\pmb{x}_i\pmb{x}_i^\top - \bar{\pmb{x}}\bar{\pmb{x}}^\top
$$(gaussian-mle)

:::{admonition} **证明**
:class: dropdown

<div style="background-color: #F8F8F8  ">

&emsp;&emsp;$N$个样本的似然函数为，

$$
\begin{split}
p(\mathcal{D})&=\prod_{i=1}^N \mathcal{N}(\pmb{x}_i|\pmb{\mu},\pmb{\Sigma})\\
&=(2\pi)^{-ND/2}|\pmb{\Sigma}|^{-N/2}\exp\left\{\sum_{i=1}^N  -\frac{1}{2}(\pmb{x}_i-\pmb{\mu})^\top\pmb{\Sigma}^{-1} (\pmb{x}_i-\pmb{\mu}) \right\}
\end{split}
$$

取对数(注意：$\pmb{\Sigma}^{-1}=\pmb{\Lambda}$)，

$$
\ell(\pmb{\mu},\pmb{\Sigma})=-\frac{ND}{2}\log (2\pi)+\frac{N}{2}\log|\pmb{\Lambda}|-\frac12\sum_{i=1}^N(\pmb{x}_i-\pmb{\mu})^\top\pmb{\Sigma}^{-1} (\pmb{x}_i-\pmb{\mu})
$$

&emsp;&emsp;先对$\pmb{\mu}$求偏导，

$$
\frac{\partial \ell}{\partial \pmb{\mu}}\xlongequal[\pmb{y}_i=\pmb{x}_i-\pmb{\mu}]{\quad\quad}\frac{\partial \ell}{\partial \pmb{y_i}}\frac{\partial \pmb{y}_i}{\partial \pmb{\mu}}=-\frac12\sum_{i=1}^N(\pmb{\Sigma}^{-1}+\pmb{\Sigma}^{-1\top})\pmb{y}_i\cdot -1
$$

令$\frac{\partial \ell}{\partial \pmb{\mu}}=0$，可解得，

$$
\hat{\pmb{\mu}}=\frac1N\sum_{i=1}^N \pmb{x}_i=\pmb{\bar{x}}
$$(multi-gauss-mle-mu)

&emsp;&emsp;下面对$\pmb{\Sigma}$求偏导，为简化计算，可转化为求$\frac{\partial \ell}{\partial \pmb{\Lambda}}$。利用迹技巧，似然函数可改写为如下形式，

$$
\begin{split}
\ell(\pmb{\Lambda})&=\frac{N}{2}\log|\pmb{\Lambda}|-\frac12\textrm{tr}\left(\underbrace{\sum_{i=1}^N(\pmb{x}_i-\pmb{\mu})(\pmb{x}_i-\pmb{\mu})^\top}_{\triangleq \pmb{S}}\pmb{\Lambda}\right)\\
&=\frac{N}{2}\log|\pmb{\Lambda}|-\frac12\textrm{tr}\left( \pmb{S}\pmb{\Lambda}\right)
\end{split}
$$

对$\pmb{\Lambda}$求偏导，可得，

$$
\frac{\partial \ell}{\partial \pmb{\Lambda}}=\frac{N}{2}\pmb{\Lambda}^{-1\top}-\frac12\pmb{S}^\top
$$

令$\frac{\partial \ell}{\partial \pmb{\Lambda}}=0$，可解出，

$$
\pmb{\Lambda}^{-1}=\frac1N\pmb{S},\quad\text{i.e.}\quad\pmb{\hat{\Sigma}}=\frac1N\sum_{i=1}^N(\pmb{x}_i-\pmb{\bar{x}})(\pmb{x}_i-\pmb{\bar{x}})^\top
$$(multi-gauss-var-mle)



</div>
:::

&emsp;&emsp;（三）**高斯判别分析**

&emsp;&emsp;MVN的一个重要应用是定义生成分类器的类条件密度函数，即

$$
p(\pmb{x}|y=c,\pmb{\theta})\triangleq\mathcal{N}(\pmb{x}|\pmb{\mu}_c,\pmb{\Sigma}_c)
$$(GDA-def)

该式称为高斯判别分析(gaussian discriminat analysis, GDA)。如果$\pmb{\Sigma}_c$为对角阵，则GDA等价于Naive Bayes。
 GDA是一种生成式方法，该方法假设数据在给定标签下服从多元高斯分布，而标签则服从伯努利分布(或Cat分布)。具体来说，样本$\pmb{x}$的条件概率$p(\pmb{x}|y=c,\pmb{\theta})$服从多元高斯分布，即$\pmb{x}\sim\mathcal{N}(\pmb{\mu},\pmb{\Sigma})$，其中$\pmb{\mu}$为均值，$\pmb{\Sigma}$为协方差矩阵。先验分布$p(y)$则服从伯努利分布(或Cat分布)。通过样本来确定高斯分布和伯努利分布(或Cat分布)的模型参数，即最大似然估计，然后通过最大后验概率来进行分类。

