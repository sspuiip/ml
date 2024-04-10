# 概率及分布

## 概率

&emsp;&emsp;频率学派认为，概率就是事件多次实验(long run)出现的次数(频率)，是一个确定的数。贝叶斯学派认为，概率是不确定性的一种量化，只与信息相关而与重复的试验不相关。

### 事件的概率

事件
: 事件定义为一个二值变量$A$，表示所描述的事件出现或不出现。

&emsp;&emsp;例如，定义事件$A$：明天将要下雨。

事件概率
: 对于一个事件出现的概率定义为：$Pr(A)$，表示事件$A$为真的概率，且满足$0\leq P(A)\leq 1$。

&emsp;&emsp;若已知事件$B$发生的前提条件下，事件$A$发生的概率怎么计算呢？很明显，这是一个条件概率。定义如下，

条件概率
: 事件$B$发生的条件下，事件$A$发生的概率，称为条件概率，记为$Pr(A|B)$。

事件独立
: 如果事件$A,B,C$满足：$Pr(A,B|C)=Pr(A|C)\cdot Pr(B|C)$，则称事件$A$与$B$条件独立，记为$A\bot B|C$。

概率空间
: 定义为一个三元组$(\Omega,\mathcal{F},\mathbb{P})$，其中$\Omega$为{样本空间}，$\mathcal{F}$为{事件空间}，即$\Omega$所有可能子集的集合，$\mathbb{P}$为一个{概率测度}即$\mathbb{P}: E\in \mathcal{F}\rightarrow [0,1]$。

Sigma-field
: 事件空间$\mathcal{F}$称之为一个$\sigma$-field，如果满足以下3个条件：
 : (1) $\emptyset\in \mathcal{F}$以及$\Omega\in \mathcal{F}$;
 : (2) $\mathcal{F}$对于补运算封闭，即$E\in \mathcal{F}$，则有$E^c\in \mathcal{F}$成立。
 : (3) $\mathcal{F}$对于有限交、并运算封闭。

Borel sigma-field
: 一个从半封闭区间$(-\infty,b]=\{x:-\infty<x\le b\le\infty\}$生成的$\sigma$-field称为**Borel sigma-field**，记为$\mathcal{B}$。通过并、交和补对这些区间运算，可以看出$\mathcal{B}$包含下面这些集合：

$$
(a,b),[a,b],(a,b],\{b\},\quad -\infty\le a\le b\le \infty
$$(borel-sigma-field)

#### 单个随机变量

随机变量
: 假设$X$为我们所感兴趣的未知数量。如，抛$n$次硬币，正面向上的次数；今天的温度等。如果$X$的值是未知且可变的，则称$X$为随机变量。

&emsp;&emsp;随机变量的取值空间称为样本空间。一个事件是样本空间中样本点的一个集合。例如，假设$X$表示抛一个色子显示的点数，则样本空间$\mathcal{X}=\{1,2,3,4,5,6\}$。某一事件$A$表示奇数点，可表示为$X\in\{1,2,3\}$；某一事件$B$表示点数介于1至3，可记为$1\leq X\leq 3$。随机变量根据样本空间的不同，分为离散型和连续型两种。

1. **离散型随机变量**

离散型随机变量
: 如果样本空间$\mathcal{X}$是有限集或无限可数集，则称$X$为离散型随机变量。用$P(X=x)$表示事件$\{X$取值$x\}$的概率。

概率质量函数(pmf)
: 随机变量取值为任意可能值的概率，称为随机变量的概率质量函数。

$$
  p(x)\triangleq Pr(X=x),\qquad \text{s.t.}\quad \left\{\begin{array}{c}
                                            0\leq p(x)\leq 1, \\
                                            \sum_x p(x)=1.
                                          \end{array}\right.
$$(discrete-pmf)




2. **连续型随机变量**

连续型随机变量
: 如果随机变量$X\in \mathbb{R}$是一个实值(real-valued)的数，则称为连续型随机变量。

&emsp;&emsp;不同于离散型随机变量，连续型随机变量不再能创建一个有限集的可能取值，即连续型随机变量在具体某一点的概率趋于0，不再具有任何意义。但是，我们可以将实线(line)划分为可数个区间，就可以将$X$与每一个区间一起来表示事件，这样又可以采用离散型随机变量的计算方法。

&emsp;&emsp;定义事件$A=(X\leq a), B=(X\leq b), C=(a<X\leq b), a<b$。显然有$B=A\vee C$，$A$与$C$相互独立，根据概率的和规则有，

$$
Pr(B)=Pr(A)+Pr(C)
$$(cont-rv)


累积分布函数
: 累积分布函数cdf为：$Pr(x)\triangleq Pr(X\leq x)$。

&emsp;&emsp;该函数为单调不减函数，则上式{eq}`cont-rv`可写成，

$$
Pr(a<X\leq b)=P(B)-P(A)
$$(cont-prob)

概率密度函数
: 定义cdf的导函数为概率密度函数(pdf)，

$$
p(x)=\frac{d}{dx}P(x)
$$(cont-pdf)

因此，等式{eq}`cont-prob`可以通过pdf改写为，

$$
Pr(a<X\leq b)=\int_a^b p(x)dx=P(B)-P(A)
$$


分位点
: 设$P$为$X$的cdf，则$P^{-1}(q)=x_q$称为分布$P$的$q$分位点，即$Pr(X\leq x_q)=q$。

#### 多个随机变量

多个随机变量的之间会有什么样的联系和特点呢？

联合分布
: 假设有$N$（不失一般性假设$N=2$）个随机变量$X,Y$，对于随机变量的任何可能值$x\in\mathcal{X},y\in\mathcal{Y}$,它们的联合分布定义为，

$$
p(x,y)\triangleq p(X=x,Y=y)
$$(joint-dist)

如下表所示：

:::{table} 联合分布
:width: 300px
:align: center
:widths: 33,33,33
| $p(X,Y)$ | $Y=0$ |$Y=1$ |
|:--: |:--: | :--:|
| $X=0$ | 0.2 |0.3 |
| $X=1$ | 0.2 |0.3 |
:::

边缘分布
: 给定联合分布，则随机变量的边缘分布为，

$$
p(X=x)=\sum_{y}p(X=x,Y=y)
$$(edge-dist-disc)

或，

$$
p(x)=\int_y p(x,y)
$$(edge-dist-cont)

如下表所示，

:::{table} 边缘分布
:width: 600px
:align: center
:widths: 30,20,20,30
| $p(X,Y)$ | $Y=0$ |$Y=1$ |边缘分布$p(X)$ |
|:--: |:--: | :--:| :--:|
| $X=0$ | 0.2 |0.3 | $p(X=0)=0.5$|
| $X=1$ | 0.2 |0.3 | $p(X=1)=0.5$|
|边缘分布$p(Y)$  | $p(Y=0)=0.4$|$p(Y=1)=0.6$ | 1|
:::


条件分布
: 给定联合分布，则随机变量的条件分布为，

$$
p(Y=y|X=x)=\frac{p(X=x,Y=y)}{p(X=x)}
$$(cond-dist-disc)

或，

$$
p(y|x)=\frac{p(x,y)}{p(x)}
$$(cond-dist-cont)


独立
: 如果$p(x,y)=p(x)p(y)$或$p(X=x,Y=y)=p(X=x)p(Y=y)$成立，则$X,Y$相互独立,记为$X\bot Y$。 如果$p(x,y|z)=p(x|z)p(y|z)$或$p(X=x,Y=y|Z=z)=p(X=x|Z=z)p(Y=y|Z=z)$成立，则$X,Y$条件独立，记为$X\bot Y|Z$。

### 随机变量概率分布的数字特征

#### 分布的矩

均值
: 对于随机变量$X$，其均值记为$\mathbb{E}[X]=\mu$，即，

$$
\begin{split}
   \mathbb{E}[X] &\triangleq \int_{x\in\mathcal{X}} x\cdot p(x)dx \\
     &\triangleq \sum_{x\in\mathcal{X}}x\cdot p(x)
\end{split}
$$(equ_dist_mean)

均值的性质
: 1. $\mathbb{E}[aX+b]=a\mathbb{E}[X]+b$
: 2. $\mathbb{E}[\sum_{i}X_i]=\sum_i \mathbb{E}[X_i]$
: 3. $\mathbb{E}[\prod_{i}X_i]=\prod_{i}\mathbb{E}[X_i]$，如果$X_i$相互独立。

方差
: 随机变量$X$的方差为$\mathbb{V}[X]$，记为$\sigma^2$。

$$
\begin{split}
\mathbb{V}[X]&\triangleq\mathbb{E}[(x-\mu)^2]\\
&=\int(x-\mu)^2p(x)dx\\
&=\int x^2p(x)dx-2\mu\int xp(x)dx+\mu^2\int p(x)dx\\
&=\mathbb{E}[X^2]-\mu^2
\end{split}
$$(equ-dist-var)

&emsp;&emsp;易知2阶矩$\mathbb{E}[X^2]=\mu^2+\sigma^2$。以及标准差$\text{std}[X]\triangleq \sqrt{\mathbb{V}[X]}=\sigma$。

方差的性质
: 1. $\mathbb{V}[aX+b]=a^2\mathbb{V}[X]$。
: 2. $\mathbb{V}[\sum_i X_i]=\sum_i \mathbb{V}[X_i]$，如果$X_i$相互独立。
: 3. $\mathbb{V}[\prod_i X_i]=\prod_{i}(\mu_i^2+\sigma_i^2)-\prod_i\mu_i^2$。

分布的众数
: 众数为概率分布的最高概率质量或概率密度的随机变量的值，即，{math}`x^*=\arg\max_x p(x)`。如果分布是多众数的，则众数不唯一。


条件矩(conditional moments)
: 当有多个依赖关系的随机变量时，可以计算某个随机变量在给定其它随机变量条件的矩。

1. law of iterated expectation: (law of total expectation)
```{math}
\begin{split}
        \mathbb{E}[\mathbb{E}[X|Y]]&=\sum_y\left(\sum_x xp(X=x|Y=y)\right)p(Y=y)\\
        &=\sum_y\sum_x xp(X=x,Y=y)\\
        &=\mathbb{E}[X]
    \end{split}
``` 

2. law of total variance: 
```{math}
 \begin{split}
     \mathbb{V}[X]&=\mathbb{E}[\mathbb{E}[X^2|Y]]-\mathbb{E}[\mathbb{E}[X|Y]]^2\\
     &=\mathbb{E}[\mathbb{V}(X|Y)] +\mathbb{E}[\mathbb{E}[X|Y]^2]-\mathbb{E}[\mathbb{E}[X|Y]]^2 \\
     &=\mathbb{E}[\mathbb{V}[X|Y]]+\mathbb{V}[\mathbb{E}[X|Y] \\
  \end{split}
```

### 常见分布

#### 伯努利分布与二项分布

&emsp;&emsp;(一) **伯努利分布**

$$
Y\sim \text{Ber}(y|\theta)=\left\{\begin{array}{ll}
                             \theta, & Y=1, \\
                             1-\theta, & Y=0.
                           \end{array}\right.
$$(ber-dist)

&emsp;&emsp;(二) **二项分布**

$$
Y\sim\text{Bin}(y|N,\theta)\triangleq\begin{pmatrix}
                                       N \\
                                       y
                                     \end{pmatrix}\theta^y(1-\theta)^{N-y}
$$(bin-dist)

&emsp;&emsp;当我们想要预测$y\in\{0,1\}$时，给定输入$x\in\mathcal{X}$，可以使用如下形式的条件概率分布，

$$
p(y|\pmb{x},\pmb{\theta})=\text{Ber}(y|f(\pmb{x},\pmb{\theta}))
$$

其中，$f$是某一函数用于预测输出变量分布的函数，函数可以有很多种类，但要满足限制$0\leq f(\pmb{x},\pmb{\theta})\leq 1$。为了避免$f$受到限制，可以使用$\sigma(\cdot)$函数，即，

$$
p(y|\pmb{x},\pmb{\theta})=\text{Ber}\{y|\sigma[f(\pmb{x},\pmb{\theta})]\}=\underbrace{\frac{1}{1+e^{-\pmb{\theta}^\top \pmb{x}}}}_{例如：f(\pmb{x},\pmb{\theta})=\pmb{\theta}^\top\pmb{x}}
$$

其中，$\sigma(\cdot)$为sigmoid函数或logistic函数。

logistic函数
: logistic函数为,

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$(logistic-fun)

logistic函数性质
: 1. $\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{e^x+1}$。
: 2. $\frac{d}{dx}\sigma(x)=\sigma(x)\cdot(1-\sigma(x))$。
: 3. $1-\sigma(x)=\sigma(-x)$。
: 4. $\sigma^{-1}(p)=\log\frac{p}{1-p}\triangleq \text{logit}(p)=x$，即，**$\sigma()$与$\text{logit}()$互为反函数**。

logistic regression
: 当$f(\pmb{x};\pmb{\theta})=\pmb{w}^\top\pmb{x}$，称为logistic regression，即，

$$
p(y=1|\pmb{x},\pmb{w})=\sigma(\pmb{w}^\top\pmb{x}+b)=\frac{1}{1+e^{-(\pmb{w}^\top\pmb{x}+b)}}
$$(logistic-regression)


#### Categorical分布与Multinomial分布 

&emsp;&emsp;(一) **Categorical分布**

&emsp;&emsp;伯努力分布是抛一次硬币事件的描述，当事件不止2个结果时，可以使用Categorical分布来描述该过程。例如：抛一次色子的点数分布（6个结果）。

Categorical分布（例如：抛1次色子的点数分布）
: 定义为，

$$
\text{Cat}(y|\pmb{\theta})\triangleq \prod_c \theta_c^{\mathbb{I}(y=c)}
$$(cat-dist)

经过one-hot编码后，可以写为，

$$
\text{Cat}(\pmb{y}|\pmb{\theta)}\triangleq \prod_c \pmb{\theta}_c^{{y}_c}
$$(cat-dist-onehot)

&emsp;&emsp;(二) **Multinomial分布**

&emsp;&emsp;Multinomial分布是Categorical分布重复$n$次。例如：抛$N$次色子的点数分布。

Multinomial分布
: 定义为，

$$
\text{Mu}(\pmb{S}|N,\pmb{\theta})\triangleq \begin{pmatrix}
                                            N \\
                                            S_1,S_2,...,S_C
                                          \end{pmatrix}\prod_c^C\pmb{\theta}_c^{S_c}
$$(multi-dist)

其中，$\pmb{S}=\sum_n \pmb{y}_n, S_c=\sum_{n=1}^N\mathbb{I}(\pmb{y}_n=c)$。

Softmax函数
: 假设有，

$$
p(\pmb{y}|\pmb{x,\theta})=\text{Cat}(\pmb{y}|f(\pmb{x,\theta}))
$$

其中$0\leq f_c(\pmb{x,\theta})\leq 1, \sum_c f_c(\pmb{x,\theta})=1$。为避免$f$直接预测概率向量，可以将$f$的输出映射至Softmax函数，也称为Multinomial logit，即

$$
\mathcal{S}(\pmb{a})\triangleq \begin{bmatrix}
                                \frac{e^{a_1}}{\sum_{c}^{C}e^{a_{c}}}, & \cdots, & \frac{e^{a_c}}{\sum_{c}^{C}e^{a_{c}}}
                              \end{bmatrix}
$$(softmax-def)

其中，$\pmb{a}=f(\pmb{x,\theta})$称为logits，是log odds的泛化。Softmax函数这样称呼主要是与argmax函数相似。

多类别logistic回归
: 假设$f(\pmb{x,\Theta})=\pmb{\Theta} \pmb{x}+\pmb{b},\pmb{\Theta}\in\mathbb{R}^{C\times D},\pmb{b}\in\mathbb{R}^C$，则模型，

$$
p(y|\pmb{x,\Theta})=\text{Cat}(y|\mathcal{S}(\pmb{\Theta} \pmb{x}+b))
$$

称为multinomial logistic regression。相应地，

$$
p({y}=c|\pmb{x,\Theta})=\frac{e^{(\pmb{\Theta x}+\pmb{b})_c}}{\sum_{c'}e^{(\pmb{\Theta x}+\pmb{b})_{c'}}}
$$(multi-logistic-reg)

log-sum-exp 技巧
: $\exp(x)$函数当$x$超出某个范围时会发生溢出，如$\exp(1000)=\infty$。使用log-sum-exp技巧可以避免这个问题，即

$$
\log\sum_c^C e^{a_c}=m+\log\sum_c^C e^{(a_c-m)}
$$(log-sum-exp)

#### 一维高斯分布

累积分布函数cdf
: 定义为，

$$
 \Phi(y|\mu,\sigma^2)&=\int_{-\infty}^{y}\mathcal{N}(z|\mu,\sigma^2)dz\\
   &=\frac{1}{2}(1+\text{erf}(z/\sqrt{2}))
$$(norm-pdf)

其中，$z=(y-\mu)/\sigma$，$\sigma^2$为方差，$\sigma^{-2}$为精度；以及erf函数，

$$
 \text{erf}(u)\triangleq \frac{2}{\sqrt{\pi}}\int_0^u e^{-t^2}dt
$$(erf-def)

概率密度函数pdf
: 定义为，

$$
\mathcal{N}(z|\mu,\sigma^2)\triangleq \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(z-\mu)^2}{2\sigma^2}}
$$(norm-pdf-1)

高斯分布性质
: $\mathbb{E}[z]=\int_{-\infty}^{+\infty}y\mathcal{N}(z|\mu,\sigma^2)dz=\mu$.
: $\mathbb{V}[z]=\mathbb{E}[(z-\mu)^2]=\int (z-\mu)^2p(z)dz=\sigma^2=\mathbb{E}[z^2]-\mathbb{E}[z]^2$.
: $\mathbb{E}[z^2]=\mathbb{E}[z]^2+\mathbb{V}[z]=\mu^2+\sigma^2$.

Probit函数
: 记高斯分布CDF逆函数为$P^{-1}(q)=y_q$，其满足$P(Y\leq y_q)=q$。标准高斯CDF逆函数记为$\Phi^{-1}(x)$。$P^{-1}(q)$也称为**Probit**函数，其映射为$:[0,1]\rightarrow (-\infty,+\infty)$。

&emsp;&emsp;例如：$(\Phi^{-1}(0.025),\Phi^{-1}(0.0975))=(-1.96,1.96)$，若分布为$\mathcal{N}(\mu,\sigma^2)$，则95\%置信区间为$(\mu-1.96\sigma,\mu+1.96\sigma)$，经常用$(\mu-2\sigma,\mu+2\sigma)$来近似。

Dirac delta函数
: 该函数为退化的高斯概率密度函数，即$\lim_{\delta\rightarrow 0}\mathcal{N}(y|\mu,\sigma^2)\rightarrow \delta(y-\mu)$，定义如下,

$$
\delta(x)=\left\{\begin{array}{ll}
                  +\infty, & \text{if }x=0, \\
                  0,&\text{if }x\neq 0,
                \end{array} \right.\qquad \text{s.t. }\int_{-\infty}^{+\infty}\delta(x)=1.
$$(dirac-delta-fun)

具有平移特性，即，

$$
\int_{-\infty}^{\infty}f(x)\delta(x-\mu)dx=f(\mu)
$$(dirac-delta-prop)

Regression
: 模型为，

$$
p(y|\pmb{x,\theta})=\mathcal{N}(y|f_\mu(\pmb{x;\theta}),f_\sigma(\pmb{x};\pmb{\sigma})^2)
$$(gauss-regression)

&emsp;&emsp;线性回归同方差模型：

$$
p(y|\pmb{x;\theta})=\mathcal{N}(\pmb{w}^\top\pmb{x}+b,\sigma^2)
$$(gauss-lin-regression)

&emsp;&emsp;线性回归异方差模型：

$$
p(y|\pmb{x;\theta})=\mathcal{N}(\pmb{w}_\mu^\top \pmb{x}+b,\sigma_+(\pmb{w}_\sigma^\top \pmb{x}))
$$(gauss-lin-regression-hete)

其中，$\sigma_+(a)=\log(1+e^a)$为softplus函数：$\mathbb{R}\rightarrow\mathbb{R}_+$。

#### 多维高斯分布

&emsp;&emsp;**(一) 定义**

&emsp;&emsp;多维高斯分布的概率密度函数: $\pmb{y}\sim \mathcal{N}(\pmb{\mu},\pmb{\Sigma)}$，定义为

$$
\mathcal{N}(\pmb{y}|\pmb{\mu},\pmb{\Sigma})\triangleq \frac{1}{(2\pi)^{D/2}|\pmb{\Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}(\pmb{y}-\pmb{\mu})^\top\pmb{\Sigma}^{-1}(\pmb{y}-\pmb{\mu})\right\}
$$(equ_mult_gauss)

&emsp;&emsp;特例：二维高斯分布

&emsp;&emsp;**(二) 二维高斯分布pdf**

&emsp;&emsp;当$D=2$时，

$$
\pmb{y}\sim \mathcal{N}(\pmb{\mu},\pmb{\Sigma}),\pmb{\mu}\in \mathbb{R}^2,\pmb{y}\in \mathbb{R}^2
$$

其中，

$$
\pmb{\Sigma}=\begin{pmatrix}
              \sigma_1^2 & \sigma_{12}^2 \\
              \sigma_{21}^2 & \sigma_2^2
            \end{pmatrix}=\begin{pmatrix}
                            \sigma_1^2 & \rho \sigma_1\sigma_2 \\
                            \rho \sigma_1\sigma_2 & \sigma_2^2
                          \end{pmatrix},\qquad \rho\triangleq corr(X,Y)=\frac{Cov(X,Y)}{\sqrt{\mathbb{V}(X)\mathbb{V}(Y)}}=\frac{\sigma_{12}^2}{\sigma_1\sigma_2}
$$

展开2维高斯分布，可得，

$$
 p(y_1,y_2)=\frac{1}{2\pi \sigma_1\sigma_2\sqrt{1-\rho^2}}\times\exp\left\{-\frac{1}{2(1-\rho^2)} \times\left[ \frac{(y_1-\mu_1)^2}{\sigma_1^2}+\frac{(y_2-\mu_2)^2}{\sigma_2^2}-2\rho\times\frac{y_1-\mu_1}{\sigma_1}\times\frac{y_2-\mu_2}{\sigma_2}\right] \right\}
$$


&emsp;&emsp;**(三) Mahalanobis 距离**

&emsp;&emsp;定义为，

$$
\Delta(\pmb{x},\pmb{\mu})\triangleq (\pmb{x}-\pmb{\mu})^{\top}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu});\quad\pmb{\Lambda}\triangleq \pmb{\Sigma}^{-1}
$$

$\pmb{\Sigma}$可特征分解为：$\pmb{\Sigma}=\sum_{d=1}^{D}\lambda_d \pmb{u}_d\pmb{u}_d^\top$，$\lambda_d,\pmb{u}_d$分别为第$d$个特征值、特征向量。其相应的精度矩阵也可以分解为：$\pmb{\Lambda}=\sum_{d=1}^{D}\frac{1}{\lambda_d}\pmb{u}_d\pmb{u}_d^\top$。定义$z_d\triangleq \pmb{u}_d^\top (\pmb{x}-\pmb{\mu})$，则$\pmb{z}=\pmb{U}(\pmb{x}-\pmb{\mu})$。则$\Delta$可重写为，

$$
\begin{split}
   \Delta(\pmb{x},\pmb{\mu}) &\triangleq (\pmb{x}-\pmb{\mu})^{\top}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu})\\
     &=(\pmb{x}-\pmb{\mu})^{\top} \left(\sum_{d=1}^{D} \frac{1}{\lambda_d} \pmb{u}_d\pmb{u}_d^\top \right)(\pmb{x}-\pmb{\mu}) \\
     &=\sum_{d=1}^{D}\frac{z_d^2}{\lambda_d}
\end{split}
$$

&emsp;&emsp;**(四) 多维高斯联合分布**

&emsp;&emsp;假设$\pmb{y}=(\pmb{y}_1,\pmb{y}_2)\sim\mathcal{N}(\pmb{\mu},\pmb{\Sigma})$，其中均值$\pmb{\mu}$与协方差矩阵$\pmb{\Sigma}$分别为，

$$
\pmb{\mu}=\begin{pmatrix}
           \pmb{\mu}_1 \\
           \pmb{\mu}_2
         \end{pmatrix},\quad \pmb{\Sigma}=\begin{pmatrix}
                                           \pmb{\Sigma}_{11} & \pmb{\Sigma}_{12} \\
                                           \pmb{\Sigma}_{21} & \pmb{\Sigma}_{22}
                                         \end{pmatrix}
$$(multi-norm-dist)

以及精度矩阵，

$$
\begin{split}
   \pmb{\Lambda}\triangleq\pmb{\Sigma}^{-1}=\begin{pmatrix} \pmb{\Lambda}_{11} & \pmb{\Lambda}_{12} \\ \pmb{\Lambda}_{21} & \pmb{\Lambda}_{22}\end{pmatrix}&=\begin{pmatrix}(\pmb{\Sigma/\Sigma}_{22})^{-1} & -(\pmb{\Sigma/\Sigma}_{22})^{-1}\pmb{\Sigma}_{12}\pmb{\Sigma}_{22}^{-1} \\ -\pmb{\Sigma}_{22}^{-1}\pmb{\Sigma}_{21}(\pmb{\Sigma/\Sigma}_{22})^{-1} & \pmb{\Sigma}_{22}^{-1}+\pmb{\Sigma}_{22}^{-1}\pmb{\Sigma}_{21}(\pmb{\Sigma/\Sigma}_{22})^{-1}\pmb{\Sigma}_{12}\pmb{\Sigma}_{22}^{-1}\end{pmatrix}\\
    &=\begin{pmatrix}
         \pmb{\Sigma}_{11}^{-1}+\pmb{\Sigma}_{11}^{-1}\pmb{\Sigma}_{12}[\pmb{\Sigma/\Sigma}_{11}]^{-1}\pmb{\Sigma}_{21}\pmb{\Sigma}_{11}^{-1}& \pmb{-\Sigma}_{11}^{-1}\pmb{\Sigma}_{12}[\pmb{\Sigma/\Sigma}_{11}]^{-1} \\
        -[\pmb{\Sigma/\Sigma}_{11}]^{-1}\pmb{\Sigma}_{21}\pmb{\Sigma}_{11}^{-1} & [\pmb{\Sigma/\Sigma}_{11}]^{-1}
      \end{pmatrix}
\end{split}
$$(pricision-matrix)

其中，$\pmb{\Sigma/\Sigma}_{22},\pmb{\Sigma/\Sigma}_{11}$分别是矩阵$\pmb{\Sigma}$关于$\pmb{\Sigma}_{22},\pmb{\Sigma}_{11}$的Schur补(Schur complements)。则，我们可以得出以下结论。

&emsp;&emsp;**（1）边缘分布**

$$
\boxed{
\begin{split}
  p(\pmb{y}_1)&\sim\mathcal{N}(\pmb{y}_1|\pmb{\mu}_1,\pmb{\Sigma}_{11})\\
  p(\pmb{y}_2)&\sim\mathcal{N}(\pmb{y}_2|\pmb{\mu}_2,\pmb{\Sigma}_{22})
\end{split}
}
$$(multi-norm-dist-edge)

&emsp;&emsp;**（2）后验分布**

$$
\boxed{
\begin{split}
       p(\pmb{y}_1|\pmb{y}_2)&=\mathcal{N}(\pmb{y}_1|\pmb{\mu}_{1|2},\pmb{\Sigma}_{1|2})\\
       \pmb{\Sigma}_{1|2}&=\pmb{\Sigma}_{11}-\pmb{\Sigma}_{12}\pmb{\Sigma}_{22}^{-1}\pmb{\Sigma}_{21}=\pmb{\Lambda}_{11}^{-1}\\
       \pmb{\mu}_{1|2}&=\pmb{\Sigma}_{1|2}(\pmb{\Lambda}_{11}\pmb{\mu}_1-\pmb{\Lambda}_{12}(\pmb{y}_2-\pmb{\mu}_2)) \\ &=\pmb{\mu}_1+\pmb{\Sigma}_{12}\pmb{\Sigma}_{22}^{-1}(\pmb{y}_2-\pmb{\mu}_2)\\
       &=\pmb{\mu}_1-\pmb{\Lambda}_{11}^{-1}\pmb{\Lambda}_{12}(\pmb{y}_2-\pmb{\mu}_2)
    \end{split}
}
$$(equ_post_norm)

以及，

$$
\boxed{
\begin{split}
       p(\pmb{y}_2|\pmb{y}_1)&=\mathcal{N}(\pmb{y}_2|\pmb{\mu}_{2|1},\pmb{\Sigma}_{2|1})\\
       \pmb{\Sigma}_{2|1}&=\pmb{\Sigma}_{22}-\pmb{\Sigma}_{21}\pmb{\Sigma}_{11}^{-1}\pmb{\Sigma}_{12}=\pmb{\Lambda}_{22}^{-1}\\
       \pmb{\mu}_{2|1}&=\pmb{\Sigma}_{2|1}(\pmb{\Lambda}_{22}\pmb{\mu}_2-\pmb{\Lambda}_{21}(\pmb{y}_1-\pmb{\mu}_1)) \\ &=\pmb{\mu}_2+\pmb{\Sigma}_{21}\pmb{\Sigma}_{11}^{-1}(\pmb{y}_1-\pmb{\mu}_1)\\
       &=\pmb{\mu}_2-\pmb{\Lambda}_{22}^{-1}\pmb{\Lambda}_{21}(\pmb{y}_1-\pmb{\mu}_1)
    \end{split}
}
$$(equ_post_norm2)

注：$\pmb{\Sigma}_{1|2}\triangleq \text{the schur compliment of }\pmb{\Sigma}_{22}$。证明过程主要是利用分块矩阵展开后凑平方项，略。

:::{admonition} **基础知识**
:class: dropdown

<div style="background-color: #F8F8F8  ">

相关知识点：Schur complements
:  Schur complements有一些非常有用的结论，在计算中有重要作用，如计算上节中的协方差矩阵的逆。

考虑一个分块矩阵，



有用结论
: 要去掉上(下)三角，只要乘(左乘、右乘皆可)上一个单位阵(其对应的上、下三角元取值非空，与原矩阵乘积后为0)。

&emsp;&emsp;例如：

$$
\begin{split}
\begin{pmatrix} \pmb{I}&\pmb{0}\\  -\pmb{CA}^{-1}&\pmb{I}\end{pmatrix} \underbrace{\begin{pmatrix} A&B\\ C&D \end{pmatrix}}_{去掉下三角C} &= \begin{pmatrix}A & B\\ 0&D-CA^{-1}B \end{pmatrix} \\
\underbrace{\begin{pmatrix}A & B\\ 0&D-CA^{-1}B \end{pmatrix}}_{去掉上三角B} \begin{pmatrix}\pmb{I} & -\pmb{A}^{-1}\pmb{B}\\ 0&\pmb{I} \end{pmatrix} &= \begin{pmatrix}A & 0\\ 0&D-CA^{-1}B \end{pmatrix}
\end{split}
$$

以及，

$$
\begin{split}
\begin{pmatrix} \pmb{I}&-\pmb{BD}^{-1}\\  \pmb{0}&\pmb{I}\end{pmatrix} \underbrace{\begin{pmatrix} A&B\\ C&D \end{pmatrix}}_{去掉上三角B} &= \begin{pmatrix}A-BD^{-1}C & 0\\ C&D \end{pmatrix} \\
\underbrace{\begin{pmatrix}A-BD^{-1}C & 0\\ C&D \end{pmatrix}}_{去掉下三角C} \begin{pmatrix}\pmb{I} & \pmb{0}\\ -\pmb{D}^{-1}\pmb{C}&\pmb{I} \end{pmatrix} &= \begin{pmatrix}A-BD^{-1}C & 0\\ 0&D \end{pmatrix}
\end{split}
$$
</div>
:::


#### 线性高斯系统

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
   \pmb{\Sigma}_{z|y}&=(\pmb{\Sigma}_z^{-1}+\pmb{W}^\top \pmb{\Sigma}_y^{-1}\pmb{W})^{-1}=\pmb{\Lambda}_{zz}^{-1}\\
   \pmb{\mu}_{z|y}&=\pmb{\Sigma}_{z|y}(\pmb{\Lambda}_{zz}\pmb{\mu}_z-\pmb{\Lambda}_{zy}(\pmb{y}_2-(\pmb{W\mu}_z+\pmb{b})))\\
   &=\pmb{\mu}_z-\pmb{\Lambda}_{zz}^{-1}\pmb{\Lambda}_{zy}(\pmb{y}_2-(\pmb{W\mu}_z+\pmb{b}))\\
   &=\pmb{\mu}_z + \pmb{\Sigma}_{zy}\pmb{\Sigma}_{yy}^{-1}(\pmb{y}_2-(\pmb{W\mu}_z+\pmb{b}))\\
  \end{split}
}
$$



