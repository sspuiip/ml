# 贝叶斯分类

## 贝叶斯决策

- **贝叶斯决策论（Bayesian decision theory）是概率框架下决策的基本方法**。

&emsp;&emsp;假设有类型标记$\mathcal{Y}=\{c_1,...,c_n\}$，在决策过程中类别$c_j$的样本划分为类别$c_i$所产生的损失记为$\lambda_{ij}$，则将样本$\pmb{x}$分类为$c_i$产生的损失期望(expected loss)，也称为条件风险，

$$
R(c_i|\pmb{x})=\sum_{j}^N \lambda_{ij}\cdot P(c_j|\pmb{x})
$$(expected-loss)

其中，$P(c_j|\pmb{x})$为类别标记$c_j$的后验概率。分类的任务是寻找一个最优分类映射$h:\mathcal{x}\rightarrow \mathcal{Y}$，从而最小化总体风险，

$$
R(h)=\mathbb{E}_{\pmb{x}}\{R[h(\pmb{x})|\pmb{x}]\}
$$(total-risk)

如果$h$对每个样本都能最小化条件风险{eq}`expected-loss`$R[h(\pmb{x}|\pmb{x}]$，则总体风险{eq}`total-risk`也能最小化。因此就有了贝叶斯判定准则：

{attribution="Bayes decision rule"}
> 为了最小化总体风险，对每个样本都选择能使条件风险{eq}`expected-loss`最小的类别标记。

也就是，

$$
h^*(\pmb{x})=\arg\min\limits_{c\in\mathcal{Y}} R(c|\pmb{x})
$$(optimal-function)

此时，$h^*$称为贝叶斯最优分类器，对应的总体风险$R(h^*)$称为贝叶斯风险。