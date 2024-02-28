# 贝叶斯分类

## 贝叶斯决策

- 贝叶斯决策论（Bayesian decision theory）是概率框架下决策的基本方法。

&emsp;&emsp;假设有类型标记$\mathcal{Y}=\{c_1,...,c_n\}$，在决策过程中类别$c_j$的样本划分为类别$c_i$所产生的**损失**记为$\lambda_{ij}$，则将样本$\pmb{x}$分类为$c_i$产生的损失期望(expected loss)，也称为**条件风险**，

$$
R(c_i|\pmb{x})=\sum_{j}^N \lambda_{ij}\cdot P(c_j|\pmb{x})
$$(expected-loss)

其中，$P(c_j|\pmb{x})$为类别标记$c_j$的**后验概率**。分类的任务是寻找一个**最优分类映射**$h:\mathcal{x}\rightarrow \mathcal{Y}$，从而最小化**总体风险**，

$$
R(h)=\mathbb{E}_{\pmb{x}}\{R[h(\pmb{x})|\pmb{x}]\}
$$(total-risk)

如果$h$对每个样本都能最小化条件风险{eq}`expected-loss`$R[h(\pmb{x}|\pmb{x}]$，则总体风险{eq}`total-risk`也能最小化。因此就有了**贝叶斯判定准则**：

{attribution="Bayes decision rule"}
> 为了最小化总体风险，对每个样本都选择能使条件风险{eq}`expected-loss`最小的类别标记。

也就是，

$$
h^*(\pmb{x})=\arg\min\limits_{c\in\mathcal{Y}} R(c|\pmb{x})
$$(optimal-function)

此时，$h^*$称为**贝叶斯最优分类器**，对应的总体风险$R(h^*)$称为**贝叶斯风险**。

## 朴素贝叶斯分类器


&emsp;&emsp;贝叶斯公式{eq}`bayes-formula`估计后验概率的主要困难在于<font color="blue">类条件概率$P(\pmb{x}|c)$是样本的所有属性上的联合概率</font>，难以从有限的训练样本直接估计得到，

$$
\underbrace{P(c|\pmb{x})}_{\mathrm{posterior}}=\frac{\overbrace{P(c)}^{\mathrm{prior}}\times\overbrace{P(\pmb{x}|c)}^{\mathrm{likelihood}}}{\underbrace{p(\pmb{x})}_{\mathrm{evidence}}}
$$(bayes-formula)

&emsp;&emsp;对于这一问题，假设所有属性相互独立，则类条件概率可以拆分为如下形式，

$$
P(\pmb{x}|c)=\prod_{i=1}^d P(\pmb{x}_i|c)
$$(class-condition)

相应地，贝叶斯判定准则式{eq}`optimal-function`可以改写为，

$$
h_{nb}(\pmb{x})=\arg\max\limits_{c\in\mathcal{Y}}P(c)\prod_{i=1}^d P(x_i|c)
$$(naive-bayes-target)

上式{eq}`naive-bayes-target`即为**朴素贝叶斯分类器**。训练该模型就是使用训练集$D$来估计先验概率$P(c)$和类条件概率$P(x_i|c)$。

- **估计先验概率$P(c)$和类条件概率$P(x_i|c)$**

&emsp;&emsp;假设$D_c$为第$c$类样本组成的集合，若独立同分布的样本足够多，则很容易估计出类别$c$的先验概率，

$$
P(c)=\frac{|D_c|}{|D|}
$$(nb-prior)

:::{table} 朴素贝叶斯分类器参数估计
:width: 650px
:align: center

|属性类型 | 计算方法 | 说明 |
| :--: | :--- | :---|
| 离散型 |  $P(x_i\|c)=\frac{\|D_{c,x_i}\|}{\|D_c\|}$    | $D_{c,x_i}$表示$D_c$中第$i$个属性取值为$x_i$的样本集合。|
|  连续型   |$p(x_i\|c)\sim \mathcal{N}(\mu_{c,i},\sigma^2_{c,i})$      | $\mu_{c,i},\sigma^2_{c,i}$分别为$c$类样本在第$i$个属性的均值和方差。 |

:::




