# Gumbel Trick

&emsp;&emsp;Gumbel Trick 是一种概率技术，主要用于处理离散分布的采样问题，尤其在深度学习（如强化学习、变分自编码器）中解决梯度不可导的挑战。考虑如下问题：如果给定$\pmb{x}\in\mathbb{R}^n$，如果从Gumbel分布独立采样$g_1,g_2,...,g_n$个样本，并将这些样本与$\pmb{x}$相加，得到新的向量$\pmb{y}=(x_1+g_1,...,x_n+g_n)$，则$\max(\pmb{y})$的分布是什么？以及$\arg\max(\pmb{y})$的分布是什么？Gumbel Trick给出了答案。

&emsp;&emsp;由前提条件可知，

$$
\pmb{y} = \pmb{x} + \pmb{g},\quad \max\{y_1,y_2,...,y_n\} = \max_i (x_i + g_i)
$$

对上式求期望，有，

$$
\begin{split}
\mathbb{E}\max\{y_1,y_2,...,y_n\}&=\mathbb{E}\max\{x_1+g_1, x_2+g_2, ..., x_n+g_n\} \\
&=\log\left( \sum_i \exp(x_i) \right)
\end{split}
$$(gumbel-trick-expectation-)

其中，$x_i+g_i$均为Gumbel分布，且这$n$个随机变量的max函数仍为Gumbel分布，该分布的期望正是等式的最后一行所给出的表达式。从上式可以看出，Gumbel Trick将**最大值函数**的期望转化为**LogSumExp函数**，而LogSumExp函数是Softmax函数的对数形式，因此，Gumbel Trick在某种程度上实现了从**最大值函数**到**Softmax函数**的平滑过渡。


- **最大值服从Gumbel分布，最大值的位置服从多项分布**

&emsp;&emsp;Gumbel Trick的**核心**是通过引入 ‌Gumbel 分布噪声‌，将离散采样转化为可导操作，同时保持采样的随机性。假设有$(\pi_1,...,\pi_n)$为非负非全零实数，且$g_1,...,g_n$为独立随机变量且服从Gumbel(0,1)分布，则有，

$$
P\left\{ k=\arg\max_i (g_i + \log\pi_i) \right\} = \frac{\pi_k}{\sum_i \pi_i}
$$(gumbel-max-trick)

也就是说，**最大值出现的位置**$k$服从**多项分布**，其概率为 $\frac{\pi_k}{\sum_i \pi_i}$。而**最大值**服从$\textrm{Gumbel}\left( \log\left(\sum_i\pi_i\right), 1\right)$。证明过程见最后一个小节。这种方法被称为 **Gumbel-Max Trick**。

&emsp;&emsp;同理，若给定$x_1,...,x_n\in\mathbb{R}$，则有，

$$
P\left\{ k=\arg\max_i (x_i + g_i) \right\} = \frac{\exp(x_k)}{\sum_i \exp(x_i)}
$$(gumbel-max-trick-exp)


&emsp;&emsp;若需要平滑化采样(可导，也就是可反向传播)，则可以使用 **Gumbel-Softmax Trick**，其公式为：

$$
p_k = \frac{\exp\left(\frac{g_k + \log\pi_k}{\tau}\right)}{\sum_j \exp\left(\frac{g_j + \log\pi_j}{\tau}\right)}
$$(gumbel-softmax-trick)

- **相关的等式**

| 公式 | 服从分布 |
| --- | --- |
|如果$t\sim\textrm{Expo}(\lambda)$，则有$(-\log t -\gamma)$|$\sim\textrm{Gumbel}(\log\lambda - \gamma, 1)$|
|$\arg\max_i (g_i+\log\pi_i)$|$\sim\textrm{Cat}\left(\frac{\pi_k}{\sum_i \pi_i}\right)$|
|$\max_i (g_i+\log\pi_i)$|$\sim \textrm{Gumbel}\left( \log\left(\sum_i\pi_i\right), 1\right) $|
|$\mathbb{E}[\max_i (g_i+\beta x_i)]=\log\left( \sum_i \exp(\beta\cdot x_i) \right)+\gamma$ | |




- **Gumbel Trick 具体应用的两种情况**

1. Gumbel-Max Trick. 目标为从$P(X=k)=\frac{a_k}{\sum_i a_i}$采样。
    - 生成独立同分布的Gumbel噪声$g_k=-\log(-\log(u_k))$，其中$u_k \sim U(0,1)$。
    - 计算加权得分：$s_k = \log(a_k) + g_k$。
    - 选择最大得分的索引：$k^* = \arg\max_k s_k$。

2. Gumbel-Softmax Trick. 目标为从$P(X=k)=\frac{a_k}{\sum_i a_i}$采样，并且需要可导。
    - 生成Gumbel噪声：$g_k = -\log(-\log(u_k))$。
    - 计算加权得分：$s_k = \log(a_k) + g_k$。
    - 使用Softmax函数平滑化采样：$p_k = \frac{\exp(s_k/\tau)}{\sum_j \exp(s_j/\tau)}$，其中 $\tau$ 是温度参数，控制平滑程度。

:::{admonition} **示例代码**.
:class: dropdown

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits):
        if not self.training and self.hard:
            return F.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()

        gumbels = self.sample_gumbel(logits.shape).to(logits.device)
        y = (logits + gumbels) / self.tau
        y_soft = F.softmax(y, dim=-1)

        if self.hard:
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        return y_soft

# 演示反向传播
if __name__ == "__main__":
    # 创建可训练参数
    logits = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]]))
    gumbel_layer = GumbelSoftmax(tau=0.5, hard=True)

    # 前向传播
    output = gumbel_layer(logits)
    print("Forward output:\n", output)

    # 模拟损失函数
    target = torch.tensor([[0.0, 1.0, 0.0]])
    loss = F.mse_loss(output, target)

    # 反向传播
    loss.backward()
    print("\nGradients for logits:\n", logits.grad)
```
:::


## Gumbel分布

&emsp;&emsp;Gumbel分布是极值理论中的重要分布，常用于描述极端事件的分布。其概率密度函数为

$$
f(x) = \frac{1}{\beta} e^{-\left(z+e^{-z}\right)},\quad z=\frac{x-\mu}{\beta}, \quad x \in \mathbb{R}.
$$(gumbel-density)

其中，$\mu$为位置参数，$\beta$为尺度参数。Gumbel分布的分布函数为    

$$
F(x) = e^{-e^{-\left(\frac{x-\mu}{\beta}\right)}}, \quad x \in \mathbb{R}.
$$(gumbel-cdf)


- **属性**.

&emsp;&emsp;其**均值**与**方差**分别为，

$$
\mathbb{E}[X] = \mu + \beta \gamma, \quad \mathbb{V}[X] = \frac{\pi^2}{6} \beta^2,
$$(gumbel-mean-var)

其中，$\gamma$为欧拉常数。

&emsp;&emsp;若$G_1,G_2,...,G_n$为**独立同分布**的Gumbel随机变量，则其**最大值**$M=\max\{G_1,G_2,...,G_n\}$服从参数为$\mu+\beta\log n$的Gumbel分布，即

$$
M \sim \textrm{Gumbel}(\mu + \beta \log n, \beta).
$$(gumbel-max-same-distribution)

&emsp;&emsp;**平移不变性**. 若$X\sim \textrm{Gumbel}(\mu, \beta)$，则对于任意常数$c$，有，

$$
X + c \sim \textrm{Gumbel}(\mu + c, \beta).
$$(gumbel-translation)


## 几个分布的关系

### **指数分布与均匀分布**.

&emsp;&emsp;若有随机变量$x$服从参数为$\lambda$的指数分布，即

$$
\boxed{x\sim \textrm{Expo}(\lambda)}
$$

则有,

$$
f(\pmb{x})=\lambda e^{-\lambda x}(x\ge 0);\quad F(x)=1-e^{-\lambda x}
$$

指数分布的分布函数逆函数为，

$$
F^{-1}(p)=\frac{-\log(1-p)}{\lambda}, \quad p\in[0,1]
$$

注意到$p$与$1-p$为同分布，故，$\frac{-\log(p)}{\lambda}$与$\frac{-\log(1-p)}{\lambda}$同分布。因此，若$p$服从$U(0,1)$均匀分布，则有

$$
F^{-1}(p)=\frac{-\log(1-p)}{\lambda}=\boxed{\frac{-\log(p)}{\lambda}\sim \textrm{Expo}(\lambda)}
$$(exp-inv-unif-relation)


### **指数分布与Gumbel分布**.

&emsp;&emsp;如果有随机变量$t$服从参数为$\exp(x)$的指数分布，即

$$
\boxed{t\sim \textrm{Expo}(\exp(x))}
$$

则有，**$\Gamma=-(\log t + \gamma)$服从参数为$x$的Gumbel分布**，即

$$
\begin{split}
P(\Gamma \leq g) &= P(-(\log t + \gamma) \leq g) \\
&= P(t \geq e^{-(g-\gamma)}) \\
&= e^{-\lambda\cdot e^{-(g-\gamma)}} \\
&= e^{-\exp(x) \cdot e^{-(g-\gamma)}} \\
&= \boxed{e^{-\exp\left\{-[g-(x-\gamma)]\right\}} \triangleq F(g)}
\end{split}
$$(gumbel-from-exp-relation)

因此，$\Gamma$服从参数为$x-\gamma$的Gumbel分布，即  

$$
\Gamma \sim \textrm{Gumbel}(x-\gamma, 1).
$$

其概率密度函数为，

$$
f(\Gamma) = e^{-(\Gamma - (x-\gamma)) - e^{-(\Gamma - (x-\gamma))}}, \quad \Gamma \in \mathbb{R}.
$$

&emsp;&emsp;**注意**：当$\gamma=x$时，$\Gamma$服从标准Gumbel分布，即$\Gamma \sim \textrm{Gumbel}(0, 1)$。

| exponential distribution | Gumbel distribution |
|-------------------------|---------------------|
| $t\sim \textrm{Expo}(\lambda=\exp(x))$ | $-(\log t +\gamma)\sim\textrm{Gumbel}(x-\gamma, 1)$ |
| $t\sim \textrm{Expo}(\lambda=\exp(x))$ | $-(\log t +x)\sim\textrm{Gumbel}(0, 1)$ |


- **Gumbel随机变量最大值函数的分布**.

&emsp;&emsp;设有$n$个独立随机变量$G_i \sim \textrm{Gumbel}(x_i, 1)$，则其最大值函数$M=\max\{x_1,x_2,...,x_n\}$服从参数为$\log(\sum_i \exp(x_i))$的Gumbel分布，即

$$
M \sim \textrm{Gumbel}\left[\log\left(\sum_i \exp(x_i)\right), 1\right].
$$(gumbel-max-relation)

&emsp;&emsp;**解**：由分布函数法可知，

$$
\begin{split}
P(M \leq m) &= P(\max\{G_1, G_2, ..., G_n\} \leq m) \\
&= P(G_1 \leq m, G_2 \leq m, ..., G_n \leq m) \\
&= \prod_{i=1}^n P(G_i \leq m) \\
&= \prod_{i=1}^n e^{-\exp(-(m - x_i))} \\
&= e^{-\sum_{i=1}^n \exp(-(m - x_i))} \\
&= e^{-\exp(-m) \cdot \sum_{i=1}^n \exp(x_i)} \\
&= e^{-\exp\left\{-(m - \log(\sum_{i=1}^n \exp(x_i)))\right\}}\triangleq \textrm{Gumbel}(\log\left(\sum_{i=1}^n \exp(x_i)\right), 1).
\end{split}
$$

### **Gumbel分布与均匀分布**

&emsp;&emsp;综合上述关系，可知，

$$
\begin{split}
\Gamma &= -\log t - \gamma \\
&= -\log \left(\frac{-\log p}{\lambda}\right) - \gamma \\
&= -\log(-\log p) - \log \lambda - \gamma \\
&= -\log(-\log p) + x - \gamma
\end{split}
$$(gumbel-unif-relation)

注意，上式第3行的$\lambda$，若取$\lambda=\exp(x)$，则可以得到第4行的结果。

### 定理

**定理**. 设有$n$个独立随机变量$T_i\sim\textrm{expo}(\lambda_i)$，则有$V=\min\{x_1,...,x_n\}$服从参数为$\sum_i\lambda_i$的指数分布，且最小值出现的位置$i^*$服从多项分布，概率为，

$$
\pmb{\pi}=\left(\frac{\lambda_1}{\sum_i \lambda_i},...,\frac{\lambda_n}{\sum_i \lambda_i}\right)
$$

并且，$V$与位置$i^*$无关。

:::{admonition} **证明**.
:class: dropdown



&emsp;&emsp;设$V=\min\{T_1,...,T_n\}$，则$V>t$等价于$T_i>t, \forall i$，因此，

$$
\begin{split}
P(V>t)&=P(T_1>t,...,T_n>t)\\
&=\prod_{i=1}^n P(T_i>t)\\
&=\prod_{i=1}^n (1-F_i(t))\\
&=\prod_{i=1}^n e^{-\lambda_i t}\\
&=e^{-\left(\sum_{i=1}^n \lambda_i\right) t}
\end{split}
$$

因此，$V\sim \textrm{Expo}(\sum_i \lambda_i)$。又因为，$V=T_{i^*}$，则有，

$$
\begin{split}
P(i^*=k,V>t)&=P(T_k>t, T_i>T_k, \forall i\ne k)\\
&=P(T_k>t)\prod_{i\ne k} P(T_i>T_k)\\
&=P(T_k>t)\prod_{i\ne k}\int_t^{\infty} \lambda_i e^{-\lambda_i s} ds\\
&=P(T_k>t)\prod_{i\ne k} e^{-\lambda_i t}\\
&=P(T_k>t) e^{-\left(\sum_{i\ne k} \lambda_i\right) t}\\
&=e^{-\lambda_k t} e^{-\left(\sum_{i\ne k} \lambda_i\right) t}\\
&=e^{-\left(\sum_{i=1}^n \lambda_i\right) t}
\end{split} 
$$

因此，$P(i^*=k|V>t)=\frac{\lambda_k}{\sum_i \lambda_i}$，且与$t$无关。证毕。
:::

### 推论

**推论**. 设有$n$个独立随机变量$\Gamma_i\sim\textrm{Gumbel}(x_i,1)$，则有$M=\max\{\Gamma_1,...,\Gamma_n\}$服从参数为$\log(\sum_i \exp(x_i))$的Gumbel分布，且最大值出现的位置$i^*$服从多项分布，概率为，

$$
\pmb{\pi}=\left(\frac{\exp(x_1)}{\sum_i \exp(x_i)},...,\frac{\exp(x_n)}{\sum_i \exp(x_i)}\right)
$$

:::{admonition} **证明**.
:class: dropdown

&emsp;&emsp;**1. 最大值 $M$ 的分布证明**. Gumbel的CDF性质.$\Gamma_i$ 的累积分布函数（CDF）为：

$$
P(\Gamma_i \leq y) = \exp\left(-\exp\left(-(y - x_i)\right)\right).
$$

独立变量的最大值CDF. 由独立性：

$$
\begin{aligned} P(M \leq y) &= \prod_{i=1}^n P(\Gamma_i \leq y) \\ 
&=\exp\left(-\sum_{i=1}^n \exp\left(-(y - x_i)\right)\right) \\ 
&=\exp\left(-\exp(-y) \cdot S\right), \quad \boxed{S = \sum_{i=1}^n \exp(x_i)}. \end{aligned}
$$

标准化为Gumbel分布. 令 $\mu = \log S$，则：

$$
P(M \leq y) = \exp\left(-\exp\left(-(y - \mu)\right)\right),
$$

即 $M \sim \textrm{Gumbel}(\log S, 1)$。

&emsp;&emsp;**2. 最大值位置 $i^*$ 的分布证明**.核心思想：$i^* = k$ 当且仅当 $\Gamma_k > \Gamma_j$ 对所有 $j \neq k$ 成立。条件概率分解. 固定 $\Gamma_k = y$，其他 $\Gamma_j$ 需小于 $y$：

$$
P(i^* = k \mid \Gamma_k = y) = \prod_{j \neq k} P(\Gamma_j \le y) = \exp\left(-\sum_{j \neq k} \exp(-(y - x_j))\right).
$$

联合概率密度积分.结合 $\Gamma_k$ 的密度 $f_k(y) = \exp\left(-(y - x_k) - \exp(-(y - x_k))\right)$：

$$
\begin{aligned} P(i^* = k) &= \int_{-\infty}^\infty P(i^* = k \mid \Gamma_k = y) f_k(y) dy \\
&=\int_{-\infty}^\infty \exp\left(-(y - x_k) - \exp(-y)S \right) dy. \end{aligned}
$$

变量替换与积分结果. 令 $t = \exp(-y)$，则：

$$
P(i^* = k) = \exp(x_k) \int_0^\infty e^{-S t} dt = \frac{\exp(x_k)}{S}.
$$

因此 $i^* \sim \textrm{Multinomial}(\pmb{\pi})$。

:::