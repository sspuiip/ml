# 共轭分布

## Gamma分布

### Gamma函数

&emsp;&emsp;**Gamma函数**是指形如以下的函数，

$$
\Gamma(x)=\int_0^{\infty} t^{x-1}e^{-t}dt
\tag{1}
$$

该函数有如下特点：

- **递推性**. $\Gamma(x)=(x-1)\Gamma(x-1)$; 易知，$\Gamma(x) $实际上是阶乘在实数集上的扩展，即$\Gamma(x)=(x-1)!, \forall x\in\mathbb{R}$.
- **特殊点**. $\Gamma(\frac12)=\pi^{1/2}$.

### Gamma分布

&emsp;&emsp;Gamma函数在统计中出现频率很高，比如常见的三大分布($t$分布，$\chi^2$分布，$F$分布)、Beta分布、Dirichlet分布等分布的概率密度函数中都有Gamma函数。最直接相关的则是Gamma分布，即  

$$
\mathrm{Ga}(x|\alpha)=\frac{x^{\alpha-1}e^{-x}}{\Gamma(\alpha)}
\tag{2}
$$

由$\int_0^{\infty} \frac{t^{x-1}e^{-t}}{\Gamma(x)}dt=1$，易知Gamma分布的概率密度函数。


