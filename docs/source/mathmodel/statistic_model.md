### 不确定模型

&emsp;&emsp;**数学模型**是一种用变量和数学符号建立起来的等式、不等式集，用于描述客观事物的特征、内在联系的模型。若客观事物的特征具有某种不确定性，此类模型也称为**随机模型**。

&emsp;&emsp;随机变量的数字特征一般有：均值、方差、中位数、众数等。

#### 抽样分布

&emsp;&emsp;从整体$\pmb{X}$抽取一个样本$(\pmb{X}_1,...,\pmb{X}_n)$，则样本服从特定的分布，一般有：卡方分布、t分布、F分布等。

&emsp;&emsp;- **卡方分布**

&emsp;&emsp;若$\pmb{X}_i \sim \mathcal{N}(0,1)$，且$\pmb{X}_i$相互独立，则统计量

$$
\chi=\sum_i^n\pmb{X}_i^2\sim \chi^2(n)
$$

服从自由度为$n$的$\chi^2$分分布。


&emsp;&emsp;- **t分布**

&emsp;&emsp;若$\pmb{X}\sim \mathcal{N}(0,1), \pmb{Y}\sim \chi^2(n)$，则统计量，

$$
T = \frac{\pmb{X}}{\sqrt{\pmb{Y}/n}}\sim t(n)
$$

服从自由度为$n$的t分布。

&emsp;&emsp;- **F分布**

&emsp;&emsp;若$\pmb{X}\sim \chi^2(n_1), \pmb{Y}\sim \chi^2(n_2)$，则统计量，

$$
F = \frac{\pmb{X}/n_1}{\pmb{Y}/n_2}\sim F(n_1,n_2)
$$

服从第一自由度为$n_1$和第二自由度为$n_2$的F分布。

#### 参数区间估计与假设检验

&emsp;&emsp;参数的区间估计与假设检验可以看成一个问题的两个方面（都需要假设样本的分布）。**区间估计**是用统计量构建一个区间来估计未知参数，并指明此区间可以覆盖住这个参数的可靠程度。**假设检验**则利用样本数据对某个事先做出的统计假设按照某种设计好的方法（符合某种抽样分布）进行检验，判断假设是否正确。
##### $\mu$均值检测
###### 单个总体

&emsp;&emsp;若$\pmb{X}\sim \mathcal{N}(\mu,\sigma^2)$，则样本均值为$\bar{\pmb{X}}=\frac1n \sum_{i=1}^n \pmb{X}_i$，样本方差为$\pmb{S}^2=\frac{1}{n-1}\sum_{i=1}^n (\pmb{X}_i-\bar{\pmb{X}})^2$。

&emsp;&emsp;**重要结论**

&emsp;&emsp;1. $\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\sim \mathcal{N}(0,1)$，即$E[\bar{\pmb{X}}]=\mu, D[\pmb{\bar{X}}]=\sigma/\sqrt{n}, E[S^2]=\sigma^2$。

&emsp;&emsp;2. $\frac{\bar{X}-\mu}{S/\sqrt{n}}\sim \mathcal{t}(n-1)$

&emsp;&emsp;3. $\frac{(n-1)S^2}{\sigma^2}\sim \mathcal{\chi}^2(n-1)$

&emsp;&emsp;4. $\bar{\pmb{X}}$与$S^2$相互独立。


1.  **$\sigma^2$未知，使用t分布来检测$\mu$的区间**

$$
P\left\{ \left\lvert \frac{\bar{\pmb{X}}-\mu}{S/ \sqrt{n}}\right\rvert \le t_{\frac{a}{2}}(n-1)  \right\}=1-a
$$

&emsp;&emsp;即置信区间为$[\bar{\pmb{X}}-\frac{S}{\sqrt{n}}t_{\frac{a}{2}}(n-1),\bar{\pmb{X}}+\frac{S}{\sqrt{n}}t_{\frac{a}{2}}(n-1)]$

2.  **$\sigma^2$已知，使用正态分布来检测$\mu$的区间**

&emsp;&emsp;方法与t检验相似。

###### 两个总体

1.  **$\sigma_1^2=\sigma_2^2=\sigma^2$时，使用t分布来检测$\mu_1-\mu_2$的区间**

$$
\frac{(\bar{\pmb{X}}-\bar{\pmb{Y}})-(\mu_1-\mu_2)}{S_w\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}\sim t(n_1+n_2-2),\quad S_w=\frac{(n-1)S_1^2}{\sigma^2}+\frac{(n-1)S_2^2}{\sigma^2}
$$

2.  **$\sigma_1^2\neq\sigma_2^2\neq\sigma^2$时，使用t分布来检测$\mu_1-\mu_2$的区间**

$$
\frac{(\bar{\pmb{X}}-\bar{\pmb{Y}})-(\mu_1-\mu_2)}{\sqrt{\frac{S_1^2}{n_1}+\frac{S_2^2}{n_2}}}\sim t(\tilde{\nu}),\quad \tilde{\nu}=\left(\frac{S_1^2}{\sigma^2}+\frac{S_2^2}{\sigma^2}\right)^2\left/\left(\frac{(S_1^2)^2}{n_1^2(n_1-1)}+ \frac{(S_2^2)^2}{n_2^2(n_2-1)}\right)\right.
$$


##### $\sigma$方差比检测