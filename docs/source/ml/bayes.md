# 贝叶斯分类

## 贝叶斯决策

{attribution="Bayesian decision theory"}
> 贝叶斯决策论是概率框架下决策的基本方法。

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
\underbrace{P(y=c|\pmb{x})}_{\mathrm{posterior}}=\frac{\overbrace{P(y=c)}^{\mathrm{prior}}\times\overbrace{P(\pmb{x}|y=c)}^{\mathrm{likelihood}}}{\underbrace{p(\pmb{x})}_{\mathrm{evidence}}}
$$(bayes-formula)

&emsp;&emsp;对于这一问题，假设所有属性相互独立，则类条件概率可以拆分为如下形式，

$$
P(\pmb{x}|y=c)=\prod_{j=1}^d P(x_j|y=c)
$$(class-condition)

相应地，贝叶斯判定准则式{eq}`optimal-function`可以改写为，

$$
h_{nb}(\pmb{x})=\arg\max\limits_{c\in\mathcal{Y}}P(y=c)\prod_{j=1}^d P(x_j|y=c)
$$(naive-bayes-target)

上式{eq}`naive-bayes-target`即为**朴素贝叶斯分类器**。训练该模型就是使用训练集$D$来估计先验概率$P(c)$和类条件概率$P(x_i|c)$的参数。

&emsp;&emsp;（一）**模型训练**

&emsp;&emsp;单个样本的似然函数为，

$$
\begin{split}
p(\pmb{x}_i,y_i|\pmb{\theta})&=p(y=y_i|\pmb{\pi})\prod_{j=1}^D p(\pmb{x}_i|y=y_i)\\
\end{split}
$$(single-sample-likelihood)

可以得到所有样本的对数似然函数，

$$
\begin{split}
\ln p(\mathcal{D}|\pmb{\theta})&= \sum_{i=1}^N \ln p(\pmb{x}_i,y_i|\pmb{\theta})\\
&=\sum_{c=1}^C N_c\ln \pi_c +\sum_{j=1}^D\sum_{c=1}^C\sum_{i:y_i=c}\ln p(x_{ij}|\theta_{jc}),\quad N_c\triangleq\sum_{i=1}^N \mathbb{I}(y_i=c)
\end{split}
$$(dataset-log-likelihood)

&emsp;&emsp;对式{eq}`dataset-log-likelihood`求偏导，并令其等于0，可得到参数的估值，

$$
\begin{split}
\hat{\pi}_c=\frac{N_c}{N}
\end{split}
$$(pic-hat)

&emsp;&emsp;$\theta_{jc}$的估值依赖于具体的特征类型所使用的分布。**以$x_i|y\sim \text{Ber}(\theta_{jc})$为例**，

$$
\hat{\theta}_{jc}=\frac{N_{jc}}{N_c}
$$(thetajc-hat)

&emsp;&emsp;通过对上述参数的求解可以看出，该模型的训练非常容易实现，且模型训练时间复杂度仅为$O(ND)$。处理混合类型特征也相对容易实现。最大似然估计的问题是过拟合。例如，假设特征$j$的值（有且只有一个）在所有类别中出现，则可以得到$\hat{\theta}_{jc}=1$。当我们遇到一个不含有此特征值的样本时，算法将会失效，因为对于所有类来说$p(y=c|\pmb{x},\hat{\theta})=0$。一个简单的办法是贝叶斯化。

&emsp;&emsp;（二）**贝叶斯naive Bayes**

&emsp;&emsp;使用一个因子化的先验，

$$
p(\pmb{\theta})=p(\pmb{\pi})\prod_{j=1}^D\prod_{c=1}^C p(\theta_{jc}),\quad \pmb{\pi}\sim \text{Dir}(\pmb{\alpha}),\theta_{jc}\sim \text{Beta}(\beta_0,\beta_1).
$$(conjugate-prior-bayes)

与似然函数相乘后，得到后验，

$$
p(\pmb{\theta})=p(\pmb{\pi}|\mathcal{D})\prod_{j=1}^D\prod_{c=1}^C p(\theta_{jc}|\mathcal{D})
$$(posterior-bayes)

由共轭先验可知后验形式，

$$
p(\pmb{\pi}|\mathcal{D})=\text{Dir}(N_1+\alpha_1,...,N_c+\alpha_c),\quad p(\theta_{jc}|\mathcal{D})=\text{Beta}((N_c-N_{jc})+\beta_0,N_{jc}+\beta_1)
$$(posterior-bayes-detail)


&emsp;&emsp;（三）**预测**

&emsp;&emsp;预测的目标是计算后验，

$$
p(y=c|\pmb{x},\mathcal{D})\propto p(y=c)\prod_{j=1}^D p(x_j|y=c,\mathcal{D})
$$

贝叶斯的作法是把未知参数积分去除，即，

$$
\begin{split}
p(y=c|\pmb{x},\mathcal{D})&=\int \text{Cat}(y=c|\pmb{\pi})p(\pmb{\pi}|\mathcal{D})d\pi\\
&\times \prod_{j=1}^D\int \text{Ber}(x_j|y=c,\theta_{jc})p(\theta_{jc}|\mathcal{D})d\theta_{jc}\\
&=\bar{\pi}\prod_{j=1}^D\bar{\theta}_{jc}^{\mathbb{I}(x_j=1)}(1-\bar{\theta}_{jc})^{\mathbb{I}(x_j=0)}
\end{split}
$$(predict-bayes)

其中，

$$
\bar{\theta}_{jc}=\frac{N_{jc}+\beta_1}{N_c+\beta_0+\beta_1},\quad \bar{\pi}=\frac{N_c+\alpha_c}{N+\alpha_0}.\quad \alpha_0=\sum_c \alpha_c.
$$




