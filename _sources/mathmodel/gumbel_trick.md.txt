# Gumbel Trick

&emsp;&emsp;Gumbel Trick 是一种概率技术，主要用于处理离散分布的采样问题，尤其在深度学习（如强化学习、变分自编码器）中解决梯度不可导的挑战。其核心是通过引入 ‌Gumbel 分布噪声‌，将离散采样转化为可导操作，同时保持采样的随机性。Gumbel Trick 基于 ‌Gumbel 分布‌（极值分布），其累积分布函数（CDF）为：

$$
F(g)=e^{−e^{−(g−\mu)}}
$$

其中, $\mu$是位置参数，$\beta=1$。具体实现分为两步：

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


## **Gumbel分布**

&emsp;&emsp;Gumbel分布是极值理论中的重要分布，常用于描述极端事件的分布。其概率密度函数为

$$
f(x) = \frac{1}{\beta} e^{-\left(z+e^{-z}\right)},\quad z=\frac{x-\mu}{\beta}, \quad x \in \mathbb{R}.
$$(gumbel-density)

其中，$\mu$为位置参数，$\beta$为尺度参数。Gumbel分布的分布函数为    

$$
F(x) = e^{-e^{-\left(\frac{x-\mu}{\beta}\right)}}, \quad x \in \mathbb{R}.
$$(gumbel-cdf)

其均值与方差为，

$$
\mathbb{E}[X] = \mu + \beta \gamma, \quad \mathbb{V}[X] = \frac{\pi^2}{6} \beta^2,
$$(gumbel-mean-var)

其中，$\gamma$为欧拉常数。

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
F^{-1}(p)=\boxed{\frac{-\log(p)}{\lambda}\sim \textrm{Expo}(\lambda)}
$$(exp-inv-unif-relation)


### **指数分布与Gumbel分布**.

&emsp;&emsp;如果有随机变量$t$服从参数为$\exp(x)$的指数分布，即

$$
\boxed{t\sim \textrm{Expo}(\exp(x))}
$$

则有，**$\Gamma=-\log t- \gamma$服从参数为$x$的Gumbel分布**，即

$$
\Gamma=\boxed{-\log t - \gamma \sim \textrm{Gumbel}(x-\gamma,1)},\quad \mu=x-\gamma, \beta=1.
$$(gumbel-from-exp-relation)

注意：上式中$x=\log(\lambda)$。

其概率密度函数为，

$$
f(\Gamma) = e^{-(\Gamma - (x-\gamma)) - e^{-(\Gamma - (x-\gamma))}}, \quad \Gamma \in \mathbb{R}.
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
&=\exp\left(-\exp(-y) \cdot S\right), \quad S = \sum_{i=1}^n \exp(x_i). \end{aligned}
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