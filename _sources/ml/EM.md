# EM算法概述

&emsp;&emsp;概率模型一般都含有观测变量、隐变量。如果只有观测变量，则可以直接使用极大似然估计；如果模型含有隐变量时，则需要对极大似然估计法进行改进，这种改进就是EM算法。EM算法也称为含有隐变量概率模型的参数极大化似然估计法。EM算法是一种迭代算法，每次迭代由两步组成：E步，求隐变量的期望；M步，求模型的最大化。因此，这一算法也称为期望极大算法（expectation maximinzation algorithm）。

## EM算法

&emsp;&emsp;令$\pmb{x}_i$为数据$i$的可观测变量，$\pmb{z}_i$为隐变量或缺失变量，则可观测数据的最大对数似然为，

$$
\ell(\pmb{\theta})=\sum_{i=1}^N \log p(\pmb{x}_i|\pmb{\theta})=\sum_{i=1}^N\log\left[ \sum_{\pmb{z}_i}p(\pmb{x}_i,\pmb{z}_i|\pmb{\theta}) \right]
$$

很明显，这个问题难于优化，因为$\log$不能被塞进到`sum`求和运算中(包含和的对数)。EM算法绕开这个问题如下，首先定义**完全数据对数似然**（complete data log likelihood）,

$$
\ell_c(\pmb{\theta})\triangleq\sum_{i=1}^N\log p(\pmb{x}_i,\pmb{z}_i|\pmb{\theta})
$$

因为$\pmb{z}_i$未知，上式不能直接计算。可以定义**期望完全数据对数似然**(expected complete data log likelihood)，

$$
Q(\pmb{\theta},\pmb{\theta}^{t-1})=\mathbb{E}[\ell_c (\pmb{\theta}|\mathcal{D},\pmb{\theta}^{t-1})]
$$

其中$t$为当前迭代，$Q$为辅助函数(auxiliary function)，期望是关于旧参数$\pmb{\theta}^{t-1}$和观测数据$\mathcal{D}$的。E步的目标正是计算$Q$函数。M步的目标为优化$Q$函数的参数$\pmb{\theta}$，

$$
\pmb{\theta}^t =\arg\max\limits_{\pmb{\theta}} Q(\pmb{\theta},\pmb{\theta}^{t-1})
$$

最大后验则对应以下优化，

$$
\pmb{\theta}^t =\arg\max\limits_{\pmb{\theta}} Q(\pmb{\theta},\pmb{\theta}^{t-1})+\log p(\pmb{\theta})
$$

### EM算法推导

&emsp;&emsp;对于一个含有隐变量的概率模型，目标是极大化观测数据$Y$关于参数$\pmb{\theta}$的对数似然函数，即极大化下式，

$$
\ell(\pmb{\theta})=\log p(\pmb{Y}|\pmb{\theta})=\log\sum_{\pmb{Z}}p(\pmb{Y},\pmb{Z}|\pmb{\theta})
$$

上式的极大化困难主要是含有未观测变量。EM通过迭代的方法逐步近似极大化$\ell(\pmb{\theta})$，希望达到$\ell(\pmb{\theta}^t)>\ell(\pmb{\theta}^{t-1})$。现考虑，

$$
\begin{split}
\ell(\pmb{\theta})-\ell(\pmb{\theta}^{i})&=\log\left(\sum_{\pmb{Z}} p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta}) \right)-\log p(\pmb{Y}|\pmb{\theta}^i)\\
&=\log\left(\sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)   \frac{p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta})}{p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)} \right)-\log p(\pmb{Y}|\pmb{\theta}^i)\\
&\ge \sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)\log\frac{p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta})}{p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)} -\log p(\pmb{Y}|\pmb{\theta}^i)\\
&=\sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)\log\frac{p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta})}{p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)p(\pmb{Y}|\pmb{\theta}^i)}
\end{split}
$$

则有，

$$
\ell(\pmb{\theta})\ge\ell(\pmb{\theta}^{i})+\sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)\log\frac{p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta})}{p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)p(\pmb{Y}|\pmb{\theta}^i)}\triangleq B(\pmb{\theta},\pmb{\theta}^i)
$$

显然有$ \ell(\pmb{\theta}^i)=B(\pmb{\theta}^i,\pmb{\theta}^i)$成立。因此，可以使$B(\pmb{\theta},\pmb{\theta}^i)$增大的任何$\pmb{\theta}$都可以使得$\ell(\pmb{\theta})$增大。$B(\pmb{\theta},\pmb{\theta}^i)$也称之为$\ell(\pmb{\theta})$的一个下界。为此，可以极大化$B(\pmb{\theta},\pmb{\theta}^i)$，即

$$
\pmb{\theta}^{t+1}=\arg\max\limits_{\pmb{\theta}}B(\pmb{\theta},\pmb{\theta}^i)
$$

具体地，

$$
\begin{split}
\pmb{\theta}^{t+1}&=\arg\max\limits_{\pmb{\theta}}B(\pmb{\theta},\pmb{\theta}^i)\\
&=\arg\max\limits_{\pmb{\theta}}\left( \ell(\pmb{\theta}^{i})+\sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)\log\frac{p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta})}{p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)p(\pmb{Y}|\pmb{\theta}^i)}\right)\\
&=\arg\max\limits_{\pmb{\theta}}\left( \sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i)\log p(\pmb{Y}|\pmb{Z},\pmb{\theta})p(\pmb{Z}|\pmb{\theta}) \right)\\
&= \arg\max\limits_{\pmb{\theta}}\left( \sum_{\pmb{Z}} p(\pmb{Z}|\pmb{Y},\pmb{\theta}^i) p(\pmb{Y},\pmb{Z}|\pmb{\theta})  \right)\\
&=\arg\max\limits_{\pmb{\theta}} Q(\pmb{\theta},\pmb{\theta}^{t})
\end{split}
$$

到此，EM算法每次迭代都是通过不断极大化下界来极大化对数似然函数，从而实现隐变量概率模型的极大似然估计。

### 案例

&emsp;&emsp;假设有3枚硬币$A,B,C$，按以下规则投掷，并记录实验结果。先投掷$A$，若$A$的结果为正面，则投掷硬币$B$的结果记录为实验结果；若$A$的结果为反面，则投掷硬币$C$的结果记录为实验结果。假设整个实验只能观测到实验结果，不能观测到掷硬币的过程，独立重复$n$次试验，若观测到以下结果：

$$
1,1,0,1,0,0,0,1,1,0,...
$$

问这3枚硬币正面出现的概率是多少？

&emsp;&emsp;现假设硬币$A,B,C$正面向上的概率分别为$\pi,p,q$，$y$为一次实验的观测结果，$z$为$A$的投掷结果，则一次投掷的过程可以用以下模型来建模，

$$
\begin{split}
p(y|\theta)&=\sum_zp(y,z|\theta)=\sum_zp(y|z,\theta)p(z)\\
&=p(z=1)p(y|z=1)+p(z=0)p(y|z=0)\\
&=\pi p^y(1-p)^{1-y}+(1-\pi)q^y(1-q)^{1-y}
\end{split}
$$

$n$次实验结果，可以表示为，

$$
\begin{split}
\mathcal{L}(\theta)&=\prod_{j=1}^n p(y_j|\theta)\\
&=\prod_{j=1}^n \left[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}\right]
\end{split}
$$

取对数，

$$
\begin{split}
\ell(\theta)&=\sum_{j=1}^n\log p(y_j|\theta)\\
&=\sum_{j=1}^n\log \left[p(z=1)p(y|z=1)+p(z=0)p(y|z=0)\right] \\
&=\sum_{j=1}^n\log \left[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}\right]
\end{split}
$$

对该模型使用极大似然估计法，即可得最优参数值$\hat{\theta}=(\hat{\pi},\hat{p},\hat{q})$，

$$
\hat{\theta}=\arg\max\limits_{\theta}\ell(\theta)
$$

&emsp;&emsp;由于含有隐变量$z$，没有确定的观测值，上述问题没有解析解。可以通过迭代的方法求解。EM算法就是一种解决这类问题的迭代方法。

首先，对所有参数赋初值，$\theta^{(0)}=(\pi^{(0)},p^{(0)},q^{(0)})$，然后迭代计算参数的估计值，直至收敛。

- E步： 计算$\mathbb{E}_{z_j}[\log p(y_j,z_j)]$。 由题设可知(使用逆概公式)

$$
\begin{split}
p(z_j=1|y_j)&=\frac{p(y_j|z_j=1)p(z_j=1)}{\sum_{z_j} p(y|z_j)p(z_j)}\\
&=\frac{p(y_j|z_j=1)p(z_j=1)}{p(y_j|z_j=1)p(z_j=1)+p(y_j|z_j=0)p(z_j=0)}\\
&=\frac{ \pi^{(i)} {p^{(i)}}^{y_j}(1-p^{(i)})^{1-y_j} }{\pi^{(i)} {p^{(i)}}^{y_j}(1-p^{(i)})^{1-y_j}+(1-\pi^{(i)}) {q^{(i)}}^{y_j}(1-p^{(i)})
^{1-y_j}  }\\
&\triangleq\mu_j^{(i+1)}
\end{split}
$$

- M步：计算参数的新估计值

$$
\begin{split}
\pi^{(i+1)}&=\frac1n\sum_{j=1}^n\mu_j^{(i+1)}\\
p^{(i+1)}&=\frac{\sum_{j=1}^n\mu_j^{(i+1)}y_j}{\sum_{j=1}^n\mu_j^{(i+1)}}\\
q^{(i+1)}&=\frac{\sum_{j=1}^n(1-\mu_j^{(i+1)})y_j}{\sum_{j=1}^n(1-\mu_j^{(i+1)})}
\end{split}
$$

### 实验代码

```python
# -*- coding: utf-8 -*-
"""
@author: jimilaw
"""

import numpy as np


def getY():
    return np.array([1,1,0,1,0,0,1,0,1,1])

def EStep(pi,p,q,Y):
    numerator = pi*np.power(p,Y)*np.power(1-p,1-Y)
    denominator = pi*np.power(p,Y)*np.power(1-p,1-Y) + (1-pi)*np.power(q,Y)*np.power(1-q,1-Y)
    return numerator/denominator
    
   
MAX_ITERATION=1000

if __name__=="__main__":
    
    Y=getY()
    n=len(Y)
    iter = 0;
    # init param
    pi=0.5
    p=0.5
    q=0.5
    
    while iter < MAX_ITERATION:
        # 1. E步 calculate the expectation of log prob. w.r.t. p(z)
        iter = iter + 1
        Mu = EStep(pi, p, q, Y)
        
        # 2. M步 update param
        pi = np.sum(Mu)/n
        p = np.sum(Mu*Y)/np.sum(Mu)
        q = np.sum((1-Mu)*Y)/np.sum(1-Mu)
        
    print("最终解：pi=",pi,"\tp=",p,"\tq=",q)
        
```
