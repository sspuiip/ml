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

&emsp;&emsp;(一) **离散型随机变量**

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




&emsp;&emsp;(二) **连续型随机变量**

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

&emsp;&emsp;多个随机变量的之间会有什么样的联系和特点呢？

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

### 变量替换

&emsp;&emsp;(一) **离散型**

&emsp;&emsp;使用分布函数法可实现。略

$$
P_y(y)=\sum_{x:f(x)=y}P_x(x)
$$(disc-var-transfrom)

&emsp;&emsp;(二) **连续型**

&emsp;&emsp;连续型$y=h(x), x=h^{-1}(y)$也可以使用分布函数法，此外，还可以使用以下变换公式。

- 1维

$$
p_y(y)=p_x(h^{-1}(y))\cdot\left|\frac{dx}{dy}\right|
$$(1-d-trans)

- 多维

$$
 p_y(\pmb{y})=p_x(h^{-1}(\pmb{y}))\cdot\left|\begin{array}{ccc}
                                           \frac{\partial \pmb{x}_1}{\partial \pmb{y}_1} &\dots &\frac{\partial \pmb{x}_1}{\partial \pmb{y}_n}\\
                                           \frac{\partial \pmb{x}_2}{\partial \pmb{y}_1} &\dots &\frac{\partial \pmb{x}_2}{\partial \pmb{y}_n}\\
                                           \vdots&\ddots&\vdots\\
                                           \frac{\partial \pmb{x}_m}{\partial \pmb{y}_1} &\dots &\frac{\partial \pmb{x}_m}{\partial \pmb{y}_n}\\
                                         \end{array} \right|
$$(n-d-trans)

&emsp;&emsp;例如：已知gamma分布，求$y=1/x$的分布，即逆gamma分布。

$$
 \begin{split}
         p_y(y)&=\text{Ga}\left({y}^{-1}|a,b\right)\cdot|\frac{dx}{dy}|\\
         &=\frac{b^a}{\Gamma(a)}\left(\frac{1}{y}\right)^{a-1}e^{-b/y}\cdot y^{-2}\\
         &=\frac{b^a}{\Gamma(a)}y^{-(a+1)}e^{-b/y}\nonumber
      \end{split}
$$(inverse-gamma-dist)


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


#### Cat分布与多项分布 

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

#### 泊松分布

&emsp;&emsp;当随机变量取值为$X\in\{0,1,2,3,...\}$，则称$X$服从参数为$\lambda$的泊松分布，其概率质量函数pmf为，

$$
\text{Poi}(X=k|\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}
$$(poi-dist-pmf)

#### 经验分布

&emsp;&emsp;假设从某分布$p(x)$得到$N$个样本$\mathcal{D}=\{x^{(1)},x^{(2)},...,x^{(n)}\}$，我们可以使用delta函数集合来近似pdf,

$$
\hat{P}_N(x)=\frac{1}{N}\sum_{n=1}^N\delta_{x^{(i)}}(x)
$$(emp-dist)

上式{eq}`emp-dist`称为$\mathcal{D}$的经验分布。例如：$N$个数据的经验分布，其CDF为

$$
  \hat{P}_N(x)=\frac{1}{N}\sum_{n=1}^N\mathbb{I}(x^{(i)}\leq x)=\frac{1}{N}\sum_{n=1}^Nu_{x^{(i)}}(x);\quad u_y(x)=\left\{\begin{array}{ll}
                                                                 1,&y\leq x \\
                                                                 0,&y>x
                                                               \end{array} \right.
  $$


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

以二维为例，

$$
\pmb{\mu}=\begin{pmatrix}\mu_1\\ \mu_2\end{pmatrix},\quad\pmb{\Sigma}=\begin{pmatrix}
              \sigma_1^2 & \sigma_{12}^2 \\
              \sigma_{21}^2 & \sigma_2^2
            \end{pmatrix}=\begin{pmatrix}
                            \sigma_1^2 & \rho \sigma_1\sigma_2 \\
                            \rho \sigma_1\sigma_2 & \sigma_2^2
                          \end{pmatrix},\quad \rho\triangleq \text{cor}(X,Y)=\frac{Cov(X,Y)}{\sqrt{\mathbb{V}(X)\mathbb{V}(Y)}}=\frac{\sigma_{12}^2}{\sigma_1\sigma_2}
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

$$
\pmb{M}=\begin{pmatrix}
    \pmb{A} & \pmb{B} \\
    \pmb{C} & \pmb{D}
  \end{pmatrix}
$$

假设$\pmb{A},\pmb{D}$是可逆的，则我们可以得到$\pmb{M}^{-1}$，

$$
\begin{split}
  \pmb{M}^{-1}&=\begin{bmatrix}
            [\pmb{M}/\pmb{D}]^{-1} & -[\pmb{M}/\pmb{D}]^{-1}\pmb{BD}^{-1} \\
            -\pmb{D}^{-1}\pmb{C}[\pmb{M/D}]^{-1} & \pmb{D}^{-1}+\pmb{D}^{-1}\pmb{C}[\pmb{M/D}]^{-1}\pmb{BD}^{-1}
          \end{bmatrix} \\
  &=\begin{bmatrix}
     \pmb{A}^{-1}+\pmb{A}^{-1}\pmb{B}[\pmb{M/A}]^{-1}\pmb{CA}^{-1} & -\pmb{A}^{-1}\pmb{B}[\pmb{M/A}]^{-1} \\
      -[\pmb{M/A}]^{-1}\pmb{CA}^{-1} & [\pmb{M/A}]^{-1}
    \end{bmatrix}
  \end{split}
$$(inverse-square-matrix)

其中，

$$
[\pmb{M/A}]\triangleq \pmb{D}-\pmb{CA}^{-1}\pmb{B}
$$

为矩阵$\pmb{M}$关于$\pmb{A}$和$\pmb{D}$的Schur补,

 $$
 [\pmb{M/D}]\triangleq \pmb{A}-\pmb{BD}^{-1}\pmb{C}
 $$

 为矩阵$\pmb{M}$关于$\pmb{D}$的Schur补。

 此外，还有以下等式成立，

$$
\begin{split}
     [\pmb{M/D}]^{-1} &= \pmb{A}^{-1}+\pmb{A}^{-1}\pmb{B}[\pmb{M/A}]^{-1}\pmb{CA}^{-1} \\
       [\pmb{M}/\pmb{D}]^{-1}\pmb{BD}^{-1}&=\pmb{A}^{-1}\pmb{B}[\pmb{M/A}]^{-1}\\
 |\pmb{M/D}|&=|\pmb{M/A}||\pmb{D}^{-1}||\pmb{A}|
  \end{split}
$$

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


#### Wishart分布

&emsp;&emsp;Wishart分布是一种适用于正定矩阵的Gamma分布推广。经常用于协方差矩阵$\pmb{\Sigma}$或它的逆矩阵$\pmb{\Sigma}^{-1}$的不确定性建模。其pdf为，

$$
\text{Wi}(\pmb{\Sigma}|\pmb{S},\gamma)=\frac{1}{Z_{Wi}}|\pmb{\Sigma}|^{(\gamma-D-1)/2}\exp\left( -\frac12 \text{tr}(\pmb{\Sigma S}^{-1}) \right)
$$(wishart-pdf)

其中，$Z_{Wi}$是规一化因子(只存在于$\gamma > D-1$的情况)，

$$
Z_{Wi}=2^{\gamma D/2}\Gamma_D(\gamma/2)|\pmb{S}|^{-\gamma/2},\quad \Gamma_D(x)=\pi^{D(D-1)/4}\prod_{i=1}^D\Gamma(x+(1-i)/2)
$$

&emsp;&emsp;**Wishart分布与高斯分布之间有密切联系**。如果$\pmb{x}_i \sim \mathcal{N}(\pmb{0},\pmb{\Sigma})$，则散布矩阵$\pmb{S}=\sum_i^N\pmb{x}_i\pmb{x}_i^\top$服从Wishart分布$\pmb{S}\sim \text{Wi}(\pmb{\Sigma},N)$。一般来说，$\text{Wi}(\pmb{S},\gamma)$的均值和众数分别为，

$$
\text{mean}=\gamma\pmb{S},\quad \text{mode}=(\gamma-D-1)\pmb{S}
$$(wishart-mean)

&emsp;&emsp;**当$D=1$时，Wishart分布退化为Gamma分布**，即$\text{Wi}(\lambda|s^{-1},\gamma)=\text{Ga}(\lambda|\frac{\gamma}{2},\frac{1}{2s})$。

#### Inverse-Wishart分布

&emsp;&emsp;从前面的知识可知，如果$\lambda\sim\text{Ga}(a,b)$，则有$1/\lambda\sim\text{IG}(a,b)$。类似的，如果$\pmb{\Sigma}^{-1}\sim\text{Wi}(\pmb{S}^{-1},\gamma)$，则有$\pmb{\Sigma}\sim\text{IW}(\pmb{S},\gamma+D+1)$，其中$\text{IW}$为Inverse-Wishart分布，是逆Gamma分布的推广。

$$
\text{IW}(\pmb{\Sigma}|\pmb{S},\gamma)=|\pmb{\Sigma}|^{-(\gamma+D+1)/2}\exp\left(-\frac12\text{tr}(\pmb{S}\pmb{\Sigma}^{-1}) \right)
$$(inverse-gamma-pdf)

该分布的均值和众数分别为，

$$
\text{mean}=\frac{\pmb{S}}{\gamma-D-1},\quad\text{mode}=\frac{\pmb{S}}{\gamma+D+1}
$$(inverse-gamma-mean-mode)

如果$D=1$，退化为IG，即$\text{IW}(\sigma^2|s^{-1},\gamma)=\text{IG}(\gamma/2,s/2)$。

如果$s=1$，退化为inverse chi-squared分布。

#### NIW分布

&emsp;&emsp;Normal-inverse-Wishart, NIW分布的定义如下，

$$
\begin{split}
\text{NIW}(\pmb{\mu,\Sigma}|\pmb{m},\kappa,\nu,\pmb{S})&\triangleq \mathcal{N}(\pmb{\mu}|\pmb{m},\frac{1}{\kappa}\pmb{\Sigma})\times\text{IW}(\pmb{\Sigma}|\pmb{S},\nu)\\
&=\frac{1}{Z_{NIW}}|\pmb{\Sigma}|^{-1/2}\exp\left(-\frac{\kappa}{2}(\pmb{\mu}-\pmb{m})^\top \pmb{\Sigma}^{-1}(\pmb{\mu}-\pmb{m}) \right)\\
&\times |\pmb{\Sigma}|^{-(\nu+D+1)/2}\exp\left(-\frac12\text{tr}(\pmb{\Sigma}^{-1}\pmb{S}) \right)\\
\end{split}
$$

#### Student分布

&emsp;&emsp;高斯分布对异常值非常敏感，更加鲁棒的分布是Student分布，

$$
\mathcal{T}(y|\mu,\sigma^2,\nu)\propto\left[1+\frac{1}{\nu}\left( \frac{y-\mu}{\sigma}\right)^2 \right]^{-\frac{\nu+1}{2}}
$$(equ_student_dist)

:::{table} Student分布参数
:width: 300px
:align: center
:widths: 45,45
| 参数 | 解释 |
|:--: |:--: | 
|  $\mu$ | 均值 |
|  $\sigma$ | scale |
|$\nu$ | degree of freedom |
|  $\frac{\nu\sigma^2}{(\nu-2)}$ | 方差 |
:::

&emsp;&emsp;当$\nu=1$时，t分布退化为Cauchy(或Lorentz)分布

$$
\mathcal{C}(x|\mu,\gamma)=\frac{1}{\gamma^\pi}\left[1+ \left( \frac{x-\mu}{\sigma}\right)^2 \right]^{-1}
$$(equ_cauchy_dist)

&emsp;&emsp;当$\nu=1,\mu=0$时，

$$
 \mathcal{C}(x|\gamma)=\frac{2}{\gamma\pi}\left[1+ \left( \frac{x}{\sigma}\right)^2 \right]^{-1}
$$(equ_cauchy_plus)

#### Laplace分布

&emsp;&emsp;该分布的pdf为，

$$
\text{Lap}(y|\mu,b)=\frac{1}{2b}\exp\left(-\frac{|y-\mu|}{b} \right)
$$(laplace-dist)

:::{table} Laplace分布参数
:width: 300px
:align: center
:widths: 45,45
| 参数 | 解释 |
|:--: |:--: | 
|  $\mu$ | 均值 |
|  $\mu$ | mode |
|$2b^2$ |var |
:::

#### Gamma分布

&emsp;&emsp;该分布的pdf为，

$$
\text{Ga}(x|a,b)\triangleq \frac{b^a}{\Gamma(a)}x^{a-1}e^{-bx}
$$(gamma-dist)

:::{table} Gamma分布参数
:width: 300px
:align: center
:widths: 45,45
| 参数 | 解释 |
|:--: |:--: | 
|   $\frac{a}{b}$ | 均值 |
|  $\frac{a-1}{b}$ | mode |
|$\frac{a}{b^2}$ |var |
:::

&emsp;&emsp;- $a=k+1,b=1$时为泊松分布，

$$
\text{Poi}(X=k|\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\Leftrightarrow\frac{x^k e^{-x}}{\Gamma(k+1)}=\text{Ga}(x|a=k+1,b=1)
$$(poisson-dist-equ-gamma)

&emsp;&emsp;- a=1时为指数分布，

$$
\text{Expon}(x|\lambda)\triangleq \text{Ga}(x|1,\lambda)=\lambda e^{-\lambda x}
$$(exp-dist)

&emsp;&emsp;- chi-squared分布（卡方分布）,

$$
\chi_\nu^2(x)\triangleq \text{Ga}(x|\frac{\nu}{2},\frac{1}{2})
$$(chi-square-dist)

&emsp;&emsp;当$x_i\sim \mathcal{N}(0,1)$时，$\sum_i^\nu x_i^2 \sim \chi_\nu^2(x)$。

#### Beta分布

&emsp;&emsp;该分布的pdf为，

$$
 \text{Beta}(x|a,b)=\frac{1}{B(a,b)}x^{a-1}(1-x)^{b-1}
$$(equ-beta-dist)

:::{table} Beta分布参数
:width: 300px
:align: center
:widths: 45,45
| 参数 | 解释 |
|:--: |:--: | 
|  $\frac{a}{a+b}$ | 均值 |
|  $\frac{a-1}{a+b-2}$ | mode |
| $\frac{ab}{(a+b)^2(a+b+1)}$ |var |
:::


### 指数族分布

&emsp;&emsp;指数族包含了许多的概率分布，在统计与机器学习中具有非常重要的作用。考虑一族由参数$\pmb{\theta}\in \mathbb{R}^K$刻画的概率分布。指数族分布是指分布$p(\pmb{y}|\pmb{\theta})$，它的概率密度可以写成如下形式，

$$
\begin{split}
  p(\pmb{y}|\pmb{\theta})&=\frac{1}{Z(\pmb{\theta})}h(\pmb{y})\exp[\pmb{\theta}^\top \pmb{t}(\pmb{y})]\\&=h(\pmb{y})\exp[\pmb{\theta}^\top \pmb{t}(\pmb{y})-A(\pmb{\theta})]
\end{split}
$$(expon-family-dist)

其中的参数如下：

:::{table} 指数族分布参数
:width: 500px
:align: center
:widths: 33,66
| 参数 | 解释 |
|:--: |:--: | 
|  $h(\pmb{y})$ | 缩放常量，一般取值为1 |
|  $\pmb{t}(\pmb{y})$ | 充分统计量。(sufficient statistics) |
| $\pmb{\theta}$ |自然参数或规范参数(canonical parameter) |
|$Z(\pmb{\theta})$|正则化常量，也称为配分函数(partition function) |
| $A(\pmb{\theta})$|log配分函数 ,$A(\pmb{\theta})=\log Z(\pmb{\theta})$|
:::

若$\pmb{\theta}=f(\pmb{\phi})$，则，

$$
p(\pmb{y}|\pmb{\phi})=h(\pmb{y})\exp[f(\pmb{\phi})^\top \pmb{t}(\pmb{y})-A(f(\pmb{\phi}))]
$$(expon-family-dist-especil)

- 如果$f$为非线性映射，则称之为曲线指数族(curved exponential family)。

- 如果$\pmb{\theta}=f(\pmb{\phi})=\pmb{\phi}$，则称之为规范型(canonical form)。

- 如果$\pmb{t}(\pmb{y})=\pmb{y}$,则称之为自然指数族(natural exponential family)。

#### 例子

&emsp;&emsp;(一) **伯努利分布**

&emsp;&emsp;伯努利分布可以改写以如下形式，

$$
\begin{split} \text{Ber}(y|\mu)&=\mu^y(1-\mu)^{1-y}\\
  &=\exp[y\log(\mu)+(1-y)\log(1-\mu)]\\
  &=\exp[\pmb{t}(y)^\top \pmb{\theta}]
 \end{split}
$$

其中，$\pmb{t}(y)=[\mathbb{I}(y=1),\mathbb{I}(y=0)], \pmb{\theta}=[\log(\mu),\log(1-\mu)]$。这是一种过于完整的表示形式，因为特征之间存在依赖，如：

$$
\pmb{1}^\top\pmb{t}(y)=\mathbb{I}(y=1)+\mathbb{I}(y=0)=1
$$

如果这个表示形式是过于完整的，$\pmb{\theta}$将不会是唯一。这可以使用最小化表示来解决，即，

$$
\text{Ber}(y|\mu)=\exp\left[y\log\left(\frac{\mu}{1-\mu}\right)+\log(1-\mu)\right]
$$

可以将这些表示成指数族形式，即，

$$
\begin{split}
   \theta&=\log\left(\frac{\mu}{1-\mu}\right)\\
   \pmb{t}(y)&=y \nonumber \\
   A(\theta)&=-\log(1-\mu)=\log(1+e^\theta)\\
   h(y)&=1\\
\end{split}
$$

