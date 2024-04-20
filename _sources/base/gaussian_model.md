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

该式称为**高斯判别分析**(gaussian discriminate analysis, GDA)。如果$\pmb{\Sigma}_c$为对角阵，则GDA等价于Naive Bayes。GDA是一种生成式方法，该方法假设数据在给定标签下服从多元高斯分布，而标签则服从伯努利分布(或Cat分布)。具体来说，样本$\pmb{x}$的条件概率$p(\pmb{x}|y=c,\pmb{\theta})$服从多元高斯分布，即$\pmb{x}\sim\mathcal{N}(\pmb{\mu},\pmb{\Sigma})$，其中$\pmb{\mu}$为均值，$\pmb{\Sigma}$为协方差矩阵。先验分布$p(y)$则服从伯努利分布(或Cat分布)。通过样本来确定高斯分布和伯努利分布(或Cat分布)的模型参数，即最大似然估计，然后通过最大后验概率来进行分类。对于任意给定的样本$\pmb{x}$，可使用下式规则决策类别，

 $$
\hat{y}(\pmb{x})=\arg\max\limits_{c}\quad\left[\log p(y=c|\pmb{\pi})+\log p(\pmb{x}|\pmb{\theta}_c) \right]
 $$(gda-decision-rule)

当计算$\pmb{x}$的类$c$条件概率时，使用的是$\pmb{x}$与$\pmb{\mu}_c$的马氏距离。该过程也可以认为是近邻中心分类。对于上式{eq}`gda-decision-rule`使用均匀先验，则分类规则可以简化为，

$$
\hat{y}(\pmb{x})=\arg\max\limits_{c} \quad(\pmb{x}-\pmb{\mu}_c)^\top\pmb{\Sigma}_c^{-1}(\pmb{x}-\pmb{\mu}_c)
$$(gda-decision-rule-without-prior)

&emsp;&emsp;- **二次判别分析**

&emsp;&emsp;由贝叶斯公式可知类别后验为$p(y=c|\pmb{x})=\frac{p(\pmb{x}|y=c)p(y=c)}{\sum_{c'}p(\pmb{x}|y=c')p(y=c')}$，如果把类条件概率密度定义为高斯密度，则有，

$$
p(y=c|\pmb{x})=\frac{ \pi_c\cdot (2\pi)^{-D/2}\cdot |\pmb{\Sigma}_c|^{-1/2} \cdot \exp\left\{-\frac12 (\pmb{x}-\pmb{\mu}_c)^\top\pmb{\Sigma}_c^{-1}(\pmb{x}-\pmb{\mu}_c) \right\} }{   \sum_{c'}\pi_{c'}\cdot (2\pi)^{-D/2}\cdot |\pmb{\Sigma}_{c'}|^{-1/2} \cdot \exp\left\{-\frac12 (\pmb{x}-\pmb{\mu}_{c'})^\top\pmb{\Sigma}_{c'}^{-1}(\pmb{x}-\pmb{\mu}_{c'}) \right\} }
$$(quadratic-da)

上式{eq}`quadratic-da`也称为**二次判别分析**(quadratic disciminate analysis, QDA)。

&emsp;&emsp;- **线性判别分析**

&emsp;&emsp;对于二次判别分析QDA，考虑一个特殊情况$\pmb{\Sigma}_c=\pmb{\Sigma}$，即所有类条件概率的方差相等，则QDA可简化为，

$$
\begin{split}
p(y=c|\pmb{x},\pmb{\theta})&\propto \pi_c\exp\left\{ (\pmb{x}-\pmb{\mu})^\top \pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu})\right\}\\
&=\exp\left\{\pmb{\mu}_c\pmb{\Sigma}^{-1}\pmb{x}-\frac12\pmb{\mu}_c^\top\pmb{\Sigma}^{-1}\pmb{\mu}_c+\log\pi_c \right\} \cdot\underbrace{\exp\left\{ -\frac12\pmb{x}^\top\pmb{\Sigma}^{-1}\pmb{x} \right\}}_{与c无关，同时出现在分子分母会抵消}
\end{split}
$$(linear-discrimante-analysis)

记$\pmb{\beta}_c=\pmb{\Sigma}^{-1}\pmb{\mu}_c, \gamma_c=-\frac12\pmb{\mu}_c^\top\pmb{\Sigma}^{-1}\pmb{\mu}_c+\log\pi_c$，则上式可改写为，

$$
p(y=c|\pmb{x},\pmb{\theta})=\frac{e^{\pmb{\beta}_c^\top\pmb{x}+\gamma_c}}{\sum_{c'}e^{\pmb{\beta}_{c'}^\top\pmb{x}+\gamma_{c'}}}=\underbrace{\mathcal{S}(\eta)_c}_{S为Softmax函数}
$$(lda-def)

&emsp;&emsp;该式{eq}`lda-def`有一个有趣的属性，如果对分子取对数，则会得到一个关于$\pmb{x}$的线性函数，任意两类$c$与$c'$的决策边界将会是一条直线。因此该方法也称为线性判别分析(linear discriminate analysis, LDA)，**分类界线**可由下式给出，

$$
\begin{split}
p(y=c|\pmb{x},\pmb{\theta})&=p(y=c'|\pmb{x},\pmb{\theta})\\
\pmb{\beta}_c^\top\pmb{x}+\gamma_c&=\pmb{\beta}_{c'}^\top\pmb{x}+\gamma_{c'}\\
\pmb{x}^\top (\pmb{\beta}_c-\pmb{\beta}_{c'})&=\gamma_{c'}-\gamma_c
\end{split}
$$(lda-border)


&emsp;&emsp;- **线性判别分析-两个类别情况**

&emsp;&emsp;当只有两个类别的情形，

$$
p(y=1|\pmb{x},\pmb{\theta})=\frac{e^{\pmb{\beta}_1^\top\pmb{x}+\gamma_1}}{e^{\pmb{\beta}_1^\top\pmb{x}+\gamma_1}+e^{\pmb{\beta}_0^\top\pmb{x}+\gamma_0}}=\frac{1}{1+e^{-(\pmb{\beta}_1^\top\pmb{x}+\gamma_1-\pmb{\beta}_0^\top\pmb{x}-\gamma_0)}}
$$

此时，$\gamma_1-\gamma_0=-\frac{1}{2}(\pmb{\mu}_1-\pmb{\mu}_0)^\top\pmb{\Sigma}^{-1}(\pmb{\mu}_1+\pmb{\mu}_0)+\log(\pi_1/\pi_0)$。若定义

$$
\begin{split}
\pmb{w}&\triangleq \pmb{\beta}_1-\pmb{\beta}_0=\pmb{\Sigma}^{-1}(\pmb{\mu}_1-\pmb{\mu}_0)\\
\pmb{x}_0&=\frac12(\pmb{\mu}_1+\pmb{\mu}_0)-(\pmb{\mu}_1-\pmb{\mu}_0)\frac{\log(\pi_1/\pi_2)}{ (\pmb{\mu}_1-\pmb{\mu}_0)^\top\pmb{\Sigma}^{-1}(\pmb{\mu}_1-\pmb{\mu}_0) }
\end{split}
$$

则有，

$$
\pmb{w}^\top\pmb{x}_0=-(\gamma_1-\gamma_0)
$$

因此，

$$
p(y=1|\pmb{x},\pmb{\theta})=\text{sigm}(\pmb{w}^\top(\pmb{x}-\pmb{x_0}))
$$(lda-2-class)

&emsp;&emsp;最终决策规则为，移动$\pmb{x}$至$\pmb{x}_0$点，投影到直线$\pmb{w}$，观察结果为正或负，为正则判别为类1，否则判别为类0。

&emsp;&emsp;- **判别模型的参数估计**

&emsp;&emsp;已知判别模型的参数，我们可以根据模型来进行类别判定。然而，当参数未知时，我们要对这些模型参数先进行估计。最简单的方法是最大似然估计(MLE)。已有数据$\mathcal{D}$其对数似然函数为，

$$
p(\mathcal{D}|\pmb{\theta})=\left[\sum_{i=1}^N\sum_{c=1}^C \mathbb{I}(y_i=c)\log\pi_c \right]+\sum_{c=1}^C\left[\sum_{i:y_i=c}\log \mathcal{N}(\pmb{x}_i|\pmb{\mu}_c,\pmb{\Sigma}_c) \right]
$$(data-log-likelihood)

可以看到，似然函数可以划分为先验$\pmb{\pi}$的项和$C$个$\pmb{\mu}_c,\pmb{\Sigma}_c$的项，因此参数可以分别单独估计。对于类别先验可以使用$\pmb{\hat{\pi}}=\frac{N_c}{N}$估计，与naive Bayes一致。对于类条件密度参数，可以根据类别标签将数据集拆分成$C$个子数据集，然后分别估计该子集的类别参数，

$$
\pmb{\hat{\mu}}_c=\frac{1}{N_c}\sum_{i:y_i=c}\pmb{x}_i,\quad\pmb{\hat{\Sigma}}_c=\frac{1}{N_c}\sum_{i:y_i=c}(\pmb{x}_i-\pmb{\hat{\mu}}_c)(\pmb{x}_i-\pmb{\hat{\mu}}_c)^\top
$$(lda-para-esitmator)



## 线性高斯系统

&emsp;&emsp;假设$\pmb{z}\in \mathbb{R}^L$为未知向量，$\pmb{y}\in \mathbb{R}^D$，且它们之间的关系如下，

$$
\begin{split}
     p(\pmb{z}) &=\mathcal{N}(\pmb{z}|\pmb{\mu}_z,\pmb{\Sigma}_z) \\
       p(\pmb{y}|\pmb{z})&=\mathcal{N}(\pmb{y}|\pmb{Wz}+\pmb{b},\pmb{\Sigma}_y) \nonumber
  \end{split}
$$(lin-gauss)

则上式称为线性高斯系统。相应的联合分布$p(\pmb{z},\pmb{y})=p(\pmb{y}|\pmb{x})p(\pmb{x})$是一个$L+D$维的高斯分布，其均值与协方差为，

$$
\begin{split}
   \pmb{\mu} &=\begin{pmatrix}
                \pmb{\mu}_z \\
                \pmb{W\mu}_z+\pmb{b}
              \end{pmatrix} \\
     \pmb{\Sigma} &=\begin{pmatrix}
                     \pmb{\Sigma}_z & \pmb{\Sigma}_z \pmb{W}^{\top} \\
                    \pmb{W}\pmb{\Sigma}_z & \pmb{\Sigma}_y+\pmb{W}\pmb{\Sigma}_z\pmb{W}^{\top}
                   \end{pmatrix}
\end{split}
$$(equ_lin_gauss)


&emsp;&emsp;**（一）高斯配方法**

&emsp;&emsp;根据指数族分布，高斯分布$\mathcal{N}(\pmb{x}|\pmb{\mu},\pmb{\Sigma})$，

$$
\mathcal{N}(\pmb{x}|\pmb{\mu},\pmb{\Sigma})\triangleq \frac{1}{(2\pi)^{D/2}|\pmb{\Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}(\pmb{x}-\pmb{\mu})^\top\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu})\right\}
$$(multi-gauss-dist-normal)

可以写成经典型(Canonical form)，即

$$
\mathcal{N}(\pmb{x}|\pmb{\mu},\pmb{\Sigma})=\exp\left\{-\frac{1}{2}\pmb{x}^{\top}\pmb{\Sigma}^{-1}\pmb{x}+\pmb{\eta}^\top \pmb{x}+\pmb{\zeta} \right\}
$$(canonical-form-gauss)


其中，

$$
\begin{split}
\pmb{\eta}&=\pmb{\Sigma}^{-1}\pmb{\mu}\\
\pmb{\zeta}&=-\frac{1}{2}\left(d\log 2\pi -\log|\pmb{\Lambda}|+\pmb{\eta}^\top \pmb{\Lambda}^{-1}\pmb{\eta}\right)\nonumber
\end{split}
$$

将$\pmb{\zeta}$中的最后一项展开，我们可以发现，$\pmb{\eta}^\top \pmb{\Lambda}^{-1}\pmb{\eta}=\pmb{\mu}^\top \pmb{\Lambda}\pmb{\mu}$。

&emsp;&emsp;通过配方，可以得到$p(\pmb{z},\pmb{y})$如下，

$$
\begin{split}
   p(\pmb{z},\pmb{y})&=p(\pmb{z})p(\pmb{y}|\pmb{z}) \\
     &=\mathcal{N}(\pmb{\mu}_z,\pmb{\Sigma}_z)\cdot \mathcal{N}(\pmb{Wz}+\pmb{b},\pmb{\Sigma}_y)\\
     &=\exp \left( -\frac{1}{2}\pmb{z}^\top \pmb{\Sigma}_z^{-1}\pmb{z}+\pmb{\mu}^{\top}\pmb{\Sigma}_z^{-1}\pmb{z} +C_1\right) \\
     &\times\exp\left( -\frac{1}{2}\pmb{y}^{\top}\pmb{\Sigma}_y^{-1}\pmb{y}+(\pmb{Wz}+\pmb{b})^{\top}\pmb{\Sigma}_y^{-1} \pmb{y}-\frac{1}{2}(\pmb{Wz}+\pmb{b})^{\top}\pmb{\Sigma}_y^{-1}(\pmb{Wz}+\pmb{b})+C_2\right)\\
     &=\exp\left(-\frac{1}{2}\pmb{z}^\top \left[\pmb{\Sigma}_z^{-1}+\pmb{W}^\top \pmb{\Sigma}_y^{-1}\pmb{W}\right]\pmb{z}+\pmb{z}^\top \pmb{W}^\top\pmb{\Sigma}_y^{-1}\pmb{y}-\frac{1}{2}\pmb{y}^{\top}\pmb{\Sigma}_y^{-1}\pmb{y} \right)+C_3\\
     &=\exp\left(-\frac{1}{2}\begin{pmatrix}\pmb{z}\\ \pmb{y}\end{pmatrix}^\top\begin{pmatrix} \pmb{\Sigma}_z^{-1}+\pmb{W}^\top \pmb{\Sigma}_y^{-1}\pmb{W} & -\pmb{W}^\top \pmb{\Sigma}_y^{-1}\\ -\pmb{\Sigma}_y^{-1}\pmb{W}& \pmb{\Sigma}_y^{-1}                                               \end{pmatrix}\begin{pmatrix}\pmb{z}\\ \pmb{y}\end{pmatrix}+ \begin{pmatrix}
       \pmb{\eta}_z\\
       \pmb{\eta}_y
     \end{pmatrix}^\top \begin{pmatrix}
       \pmb{z}\\
       \pmb{y}
     \end{pmatrix} +C_4\right)
\end{split}
$$(equ_pzy)

其中，最后一行的$\pmb{\eta}_z=\pmb{\mu}_z^\top\pmb{\Sigma}_z^{-1}$，$\pmb{\eta}_y=(\pmb{W\mu}_z+\pmb{b})^\top (\pmb{\Sigma}_y+\pmb{W\Sigma}_z^{-1}\pmb{W}^\top)^{-1}$。由式{eq}`equ_pzy`可知联合分布$p(\pmb{z},\pmb{y})$的精度矩阵$\pmb{\Lambda}$为，

$$
\pmb{\Sigma^{-1}}=\begin{pmatrix} \pmb{\Sigma}_z^{-1}+\pmb{W}^\top \pmb{\Sigma}_y^{-1}\pmb{W} & -\pmb{W}^\top \pmb{\Sigma}_y^{-1}\\ -\pmb{\Sigma}_y^{-1}\pmb{W}& \pmb{\Sigma}_y^{-1}\end{pmatrix}\triangleq \begin{pmatrix}\pmb{\Lambda}_{zz}&\pmb{\Lambda}_{zy}\\ \pmb{\Lambda}_{yz}&\pmb{\Lambda}_{yy} \end{pmatrix}=\pmb{\Lambda}
$$(joint-precision-matrix)

根据{eq}`pricision-matrix`可知，

$$
\pmb{\Sigma}=\begin{pmatrix} \pmb{\Sigma}_z & \pmb{\Sigma}_z\pmb{W}^\top\\ \pmb{W}\pmb{\Sigma}_z& \pmb{\Sigma}_y+\pmb{W}\pmb{\Sigma}_z\pmb{W}^\top\end{pmatrix}
$$(joint-variance-matrix)


&emsp;&emsp;**（二）分布计算**

&emsp;&emsp;由配方可知，**联合分布**为，

$$
\boxed{
\begin{split}
p(\pmb{z},\pmb{y})&=\mathcal{N}(\pmb{\mu},\pmb{\Sigma})\\
\pmb{\mu}&=\begin{pmatrix}\pmb{\mu}_z \\ \pmb{W}\pmb{\mu}_z+\pmb{b} \end{pmatrix}\\
\pmb{\Sigma}&=\begin{pmatrix} \pmb{\Sigma}_z & \pmb{\Sigma}_z\pmb{W}^\top\\ \pmb{W}\pmb{\Sigma}_z& \pmb{\Sigma}_y+\pmb{W}\pmb{\Sigma}_z\pmb{W}^\top\end{pmatrix}
\end{split}
}
$$(joint-gaussian-distribution)

&emsp;&emsp;由配方可知，**边缘分布**为，

$$
\boxed{
  \begin{split}
  \pmb{y}&\sim \mathcal{N}(\pmb{W\mu}_z+\pmb{b},\pmb{\Sigma}_y+\pmb{W}\pmb{\Sigma}_z\pmb{W}^\top)\\
  \pmb{z}&\sim \mathcal{N}(\pmb{\mu}_z,\pmb{\Sigma}_z)
  \end{split}
}
$$(joint-edge-dist)

&emsp;&emsp;由配方可知，**后验分布**为，

$$
\boxed{
  \begin{split}
   p(\pmb{z}|\pmb{y})&=\mathcal{N}(\pmb{y}_{z|y},\pmb{\Sigma}_{z|y})\\
   \pmb{\Sigma}_{z|y}&=\left[\pmb{\Sigma}_z^{-1}+\pmb{W}^\top \pmb{\Sigma}_y^{-1}\pmb{W}\right]^{-1}\\
   \pmb{\mu}_{z|y}&=\pmb{\Sigma}_{z|y}(\pmb{\Lambda}_{zz}\pmb{\mu}_z-\pmb{\Lambda}_{zy}(\pmb{y}_2-(\pmb{W\mu}_z+\pmb{b})))\\
   &=\pmb{\Sigma}_{z|y}\left[\pmb{\Sigma}_z^{-1}\pmb{\mu}_z+\pmb{W}^\top\pmb{\Sigma}_y^{-1}(\pmb{y}-\pmb{b}) \right]\\
   &=\pmb{\mu}_z-\pmb{\Lambda}_{zz}^{-1}\pmb{\Lambda}_{zy}(\pmb{y}_2-(\pmb{W\mu}_z+\pmb{b}))\\
   &=\pmb{\mu}_z + \pmb{\Sigma}_{zy}\pmb{\Sigma}_{yy}^{-1}(\pmb{y}_2-(\pmb{W\mu}_z+\pmb{b}))\\
  \end{split}
}
$$(posterior-gaussian-dist)

整理后验计算公式，下式为常用计算公式，即

$$
\boxed{
  \pmb{\Sigma}_{z|y}=\left[\pmb{\Sigma}_z^{-1}+\pmb{W}^\top \pmb{\Sigma}_y^{-1}\pmb{W}\right]^{-1},\quad \pmb{\mu}_{z|y}=\pmb{\Sigma}_{z|y}\left[\pmb{\Sigma}_z^{-1}\pmb{\mu}_z+\pmb{W}^\top\pmb{\Sigma}_y^{-1}(\pmb{y}-\pmb{b}) \right]
}
$$(posterior-common-used)


&emsp;&emsp;**（三）例子**

- **标量后验**

&emsp;&emsp;假设有$N$个关于潜在变量$z$的带噪测度$y_i(i=1,...,N)$，并假设带噪测度具有固定的精度$\lambda_y=\frac{1}{\sigma^2}$，所以有如下似然，

$$
p(y_i|z)=\mathcal{N}(z,\lambda_y^{-1})
$$

可以给未知变量$z$一个高斯先验，

$$
p(z)=\mathcal{N}(\mu_0,\lambda_0^{-1})
$$

则，我们对未知变量$z$可以通过计算后验$p(z|y_1,...,y_N,\sigma^2)$得到它的一个估计。

&emsp;&emsp;假设$\pmb{y}=(y_1,...,y_N)$，$\pmb{W}=\pmb{1}_N,\pmb{\Sigma}_y^{-1}=\text{diag}(\lambda_y\pmb{I})$，则有如下形式的分布，

$$
\begin{split}
z&\sim \mathcal{N}(\mu_0,\lambda_0^{-1})\\
\pmb{y}|z&\sim \mathcal{N}(\pmb{1}_N \mu_0, \text{diag}(\lambda_y^{-1}\pmb{I}))
\end{split}
$$

联合分布为，

$$
\begin{split}
p(z,\pmb{y})&=\mathcal{N}(\pmb{\mu},\pmb{\Sigma})\\
\pmb{\mu}&=\begin{pmatrix}\mu_0 \\ \pmb{1}_Nz \end{pmatrix}\\
\pmb{\Sigma}&=\begin{pmatrix} \lambda_0^{-1}&\lambda_0^{-1}\pmb{1}_N^\top \\ \pmb{1}_N\lambda_0^{-1}&\Sigma_y+\pmb{1}_N \lambda_0^{-1}\pmb{1}_N^\top \end{pmatrix}\\
\pmb{\Lambda}&=\pmb{\Sigma}^{-1}=\begin{pmatrix} \lambda_0+N\lambda_y&-\pmb{1}_N^\top \Sigma_y^{-1} \\ -\Sigma_y^{-1}\pmb{1}_N&\Sigma_y^{-1} \end{pmatrix}\\
\end{split}
$$

后验分布为，

$$
\begin{split}
p(z|\pmb{y})&=\mathcal{N}(\pmb{\mu}_{z|\pmb{y}},\pmb{\Sigma}_{z|\pmb{y}})\\
\pmb{\Sigma}_{z|\pmb{y}}&=\pmb{\Lambda}_{zz}^{-1}=(\lambda_0+N\lambda_y)^{-1} \\
\pmb{\mu}_{z|\pmb{y}}&=\pmb{\Sigma}_{z|\pmb{y}}(\pmb{\Lambda}_{zz}\mu_z-\pmb{\Lambda}_{z\pmb{y}}(\pmb{y}-\pmb{\mu}_y))\\
&=\frac{(\lambda_0+N\lambda_y)\mu_0 + \pmb{1}_N^\top \Sigma_y^{-1}(\pmb{y}-\pmb{\mu}_y) }{\lambda_0+N\lambda_y}\\
&=\frac{(\lambda_0+N\lambda_y)\mu_0 + \pmb{1}_N^\top \Sigma_y^{-1}(\pmb{y}-\pmb{1}_N \mu_0) }{\lambda_0+N\lambda_y}\\
&=\frac{\lambda_0\mu_0 + N\lambda_y \bar{y} }{\lambda_0+N\lambda_y}
\end{split}
$$


- **向量后验**

&emsp;&emsp;未知变量给一个先验分布，

$$
\pmb{z}\sim \mathcal{N}(\pmb{\mu}_z,\pmb{\Sigma}_z)
$$

假设有$N$个关于$\pmb{z}$的测量值$\pmb{y}_i,i=1,2,...,N$，则似然函数为，

$$
p(\mathcal{D}|\pmb{z})=\prod_{i=1}^{N}\mathcal{N}(\pmb{y}_i|\pmb{z},\pmb{\Sigma}_y)=\mathcal{N}(\pmb{\bar{y}}|\pmb{\mu}_z,\frac{1}{N}\pmb{\Sigma}_y)
$$

&emsp;&emsp;注意：我们可以将$N$个观测值用它们的平均值$\bar{\pmb{y}}$以及它们的方差的$1/N$来代替。设置$\pmb{W}=\pmb{I},\pmb{b}=\pmb{0}$，根据贝叶斯规则有，

$$
p(\pmb{z}|\pmb{y}_1,...,\pmb{y}_N)=\mathcal{N}(\hat{\pmb{\mu}},\hat{\pmb{\Sigma}})
$$

其中，

$$
\hat{\pmb{\Sigma}}=(\pmb{\Sigma}_z^{-1}+N\pmb{\Sigma}_y^{-1})^{-1},\quad \hat{\pmb{\mu}}=\hat{\pmb{\Sigma}}[\pmb{\Sigma}_z^{-1}\pmb{\mu}_z+\pmb{\Sigma}_y^{-1}(N\bar{\pmb{y}})]
$$



## 高斯分布的参数推理

&emsp;&emsp;上面的内容都是高斯随机变量的分布推理，前提是参数$\pmb{\mu},\pmb{\Sigma}$已知。现在考虑如何推理这些参数本身。假设获取的数据具有形式，

$$
\pmb{x}_i\sim\mathcal{N}(\pmb{\mu},\pmb{\Sigma})
$$

且数据为全可观测没有缺失值。

&emsp;&emsp;**（一）$\pmb{\mu}$的后验**

&emsp;&emsp;数据的似然函数可以表示为，

$$
p(\mathcal{D}|\pmb{\mu})=\prod_{i=1}^N \mathcal{N}(\pmb{x}_i|\pmb{\mu},\pmb{\Sigma})=\mathcal{N}(\pmb{\bar{x}}|\pmb{\mu},\frac1N\pmb{\Sigma})
$$(data-likelihood)

若给定一个关于$\pmb{\mu}$的先验$p(\pmb{\mu})=\mathcal{N}(\pmb{\mu}|\pmb{\mu}_0,\pmb{\Sigma}_0)$，则可以推理出关于$\pmb{\mu}$的后验如下，

$$
\begin{split}
p(\pmb{\mu}|\mathcal{D},\pmb{\Sigma})&=\mathcal{N}(\pmb{\mu}_N,\pmb{\Sigma}_N)\\
\pmb{\Sigma}_N^{-1}&=\pmb{\Sigma}_0^{-1}+N\pmb{\Sigma}^{-1}\\
\pmb{\mu}_N&=\pmb{\Sigma}_N(\pmb{\Sigma}_0^{-1}\pmb{\mu}_0+\pmb{\Sigma}^{-1}(N\pmb{\bar{x}}))
\end{split}
$$(mu-posterior)

上式{eq}`mu-posterior`用到了线性高斯系统的条件分布计算公式。

&emsp;&emsp;**（二）$\pmb{\Sigma}$的后验**

&emsp;&emsp;似然函数为,

$$
p(\mathcal{D}|\pmb{\mu},\pmb{\Sigma})\propto |\pmb{\Sigma}|^{-N/2}\exp\left(-\frac12\text{tr}(\pmb{S}_\mu\pmb{\Sigma}^{-1}) \right)
$$(sigma-likielihood)

若给$\pmb{\Sigma}$一个先验分布,

$$
p(\pmb{\Sigma})=\text{IW}(\pmb{\Sigma}|\pmb{S}_0,\gamma_0)\propto |\pmb{\Sigma}|^{-(\gamma_0+D+1)/2}\exp\left(-\frac12\text{tr}(\pmb{S}_0\pmb{\Sigma}^{-1}) \right)
$$(sigma-prior)

则有$\pmb{\Sigma}$的后验如下，

$$
\begin{split}
p(\pmb{\Sigma}|\mathcal{D},\pmb{\mu})&\propto p(\pmb{\Sigma})\times p(\mathcal{D}|\pmb{\mu},\pmb{\Sigma})\\
&=|\pmb{\Sigma}|^{-N/2}\exp\left(-\frac12\text{tr}(\pmb{S}_\mu\pmb{\Sigma}^{-1}) \right)\times |\Sigma|^{-(\gamma_0+D+1)/2}\exp\left(-\frac12\text{tr}(\pmb{S}_0\pmb{\Sigma}^{-1}) \right)\\
&=|\pmb{\Sigma}|^{-(N+\gamma_0+D+1)/2}\exp\left(-\frac12\text{tr}((\pmb{S}_0+\pmb{S}_\mu)\pmb{\Sigma}^{-1}) \right)\\
&=\text{IW}(\pmb{\Sigma}|\pmb{S}_N,\gamma_N)\\
\gamma_N&=\gamma_0+N\\
\pmb{S}_N&=\pmb{S}_0+\pmb{S}_\mu
\end{split}
$$(sigma-posterior)

&emsp;&emsp;**（三）$\pmb{\mu},\pmb{\Sigma}$的后验**

$$
\begin{split}
p(\pmb{\mu,\Sigma}|\mathcal{D})&\propto p(\mathcal{D}|\pmb{\mu,\Sigma})\times p(\pmb{\Sigma})p(\pmb{\mu}|\pmb{\Sigma})\\
&=(2\pi)^{-ND/2}|\pmb{\Sigma}|^{-N/2}\exp\left(-\frac{N}{2}(\pmb{\mu}-\pmb{\bar{x}})^\top \pmb{\Sigma}^{-1}(\pmb{\mu}-\pmb{\bar{x}}) \right)\exp\left(-\frac{N}{2}\text{tr}(\pmb{\Sigma}^{-1}\pmb{S}_{\bar{x}}) \right)\\
&\times \underbrace{\text{IW}(\pmb{\Sigma}|\pmb{S}_0,\nu_0)\cdot \mathcal{N}(\pmb{\mu}|\pmb{m}_0,\frac{1}{\kappa_0}\pmb{\Sigma})}_{\text{NIW}(\pmb{\mu,\Sigma}|\pmb{\mu}_0,\kappa_0,\nu_0,\pmb{S}_0)}\\
&=\text{NIW}(\pmb{\mu,\Sigma}|\pmb{\mu}_N,\kappa_N,\nu_N,\pmb{S}_N)\\
\kappa_N&=\kappa_0+N\\
\nu_N&=\nu_0+N\\
\pmb{\mu}_N&=\frac{\kappa_0\pmb{m}_0+N\pmb{\bar{x}}}{\kappa_N}=\frac{\kappa_0}{\kappa_0+N}\pmb{m}_0+\frac{N}{\kappa_0+N}\pmb{\bar{x}}\\
\pmb{S}_N&=\pmb{S}_0+\pmb{S}_{\bar{x}}+\frac{\kappa_0 N}{\kappa_0+N}(\pmb{\bar{x}}-\pmb{m}_0)(\pmb{\bar{x}}-\pmb{m}_0)^\top\\
&=\pmb{S}_0+\pmb{S}+\kappa_0\pmb{m}_0\pmb{m}_0^\top-\kappa_N\pmb{m}_N\pmb{m}_N^\top. \quad \pmb{S}\triangleq \sum_{i=1}^N\pmb{x}_i\pmb{x}_i^\top
\end{split}
$$(mu-sigma-posterior)

&emsp;&emsp;**（四）后验的边缘分布**

&emsp;&emsp;显然，

$$
p(\pmb{\Sigma}|\mathcal{D})=\int p(\pmb{\mu,\Sigma}|\mathcal{D})d\pmb{\mu}=\text{IW}(\pmb{S}_N,\nu_N)
$$(posterior-sigma-margin)

以及均值的边缘分布，

$$
p(\pmb{\mu}|\mathcal{D})=\int p(\pmb{\mu,\Sigma}|\mathcal{D})d\pmb{\Sigma}=\mathcal{T}(\pmb{\mu}|\pmb{m}_N,\frac{\pmb{S}_N}{\kappa_N(\nu_N-D+1)},\nu_N-D+1)
$$(posterior-mean-margin)
