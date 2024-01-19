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

&emsp;&emsp;Gamma函数在统计中出现频率很高，比如常见的三大分布($t$分布，$\chi^2$分布，$F$分布)、Beta分布、Dirichlet分布等分布的概率密度函数中都有Gamma函数。最直接相关的则是Gamma分布，其概率密度函数为  

$$
\mathrm{Ga}(x|\alpha)=\frac{x^{\alpha-1}e^{-x}}{\Gamma(\alpha)}
\tag{2}
$$

由$\int_0^{\infty} \frac{t^{x-1}e^{-t}}{\Gamma(x)}dt=1$，易知上式即为Gamma分布的概率密度函数。

&emsp;&emsp;令$x=\beta t$，则可以得到更一般形式的Gamma分布，

$$
\mathrm{Ga}(t|\alpha,\beta)=\frac{\beta^\alpha t^{\alpha-1} e^{-\beta t}}{\Gamma(\alpha)}\tag{3}
$$

其中，$\alpha$称为shape参数，主要决定了分布曲线的形状；$\beta$称为rate参数，主要决定曲线的陡峭。

&emsp;&emsp;**1. Gamma分布与Poisson分布**

&emsp;&emsp;参数为$\lambda$的Poissson分布，其分布律为，

$$
\mathrm{Poi}(X=k|\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\tag{4}
$$

若取$\alpha=k+1, \beta=1$，则有，

$$
\mathrm{Ga}(x|\alpha=k+1,\beta=1)=\frac{x^k e^{-x}}{\Gamma(k+1)}=\frac{x^k e^{-x}}{k!}=\mathrm{Poi}(X=k|\lambda=x)
$$

由此可见，两个分布形式上的一致性。只不过Poisson分布是离散的，Gamma分布是连续的。直观上可以认为Poisson分布是Gamma分布的特例——离散版本。

## Beta分布

&emsp;&emsp;假设均匀地从$[0,1]$随机抽取10个数字，那么第8大的数是哪个数，偏离程度小于0.01就算合格。那么该如何猜这个数字呢？如果对随机抽取的10个数字分别用随机变量$X_1,...,X_{10}$来表示，排序后可得到顺序统计量$X_{(1)},...,X_{(10)}$。一个可行的办法就是求出$X_{(8)}$的概率密度，然后使用概率分布的极值点来猜测。

&emsp;&emsp;一般来说，$X_{(k)}$的分布是什么呢？可以先尝试计算$X_{(k)}$落在区间$[x,x+\delta x]$的概率，也就是

$$
P(x\le X_{(k)}\le x+\delta x)
$$

由于在区间$[0,1]$取数，可以把该区间分割为3个部分：$[0,x),[x,x+\delta x],(x+\delta x,1]$。先考虑只有一个数落在$[x,x+\delta x]$，则$[0,x)$落入了$k-1$个数，$(x+\delta x,1]$落入了$n-k$个数。考虑以下事件，

$$
\begin{split}
E=&\{ X_1\in [x,x+\delta x],\\
&X_i \in [0,x)\quad (i=2,...,k),\\
&X_j \in (x+\delta x,1]\quad (j=k+1,...,n)\}
\end{split}
$$

则有，

$$
\begin{split}
P(E)&=\prod_{i=1}^n P(X_i)\\
&=x^{k-1}(\delta x )\underbrace{(1-x-\delta x)^{n-k}}_{\mathrm{二项式定理展开}}\\
&=x^{k-1}(1-x)^{n-k}(\delta x) + o(\delta x)
\end{split}
$$

其中$o(\delta x)$为$\delta x$的高阶无穷小。因为从$n$个数中取一个数的取法有$n$种，余下的$n-1$个数中取$k-1$个数有$\begin{pmatrix} n-1\\ k-1\end{pmatrix}$取法，所以，与$E$等价的事件有$n\begin{pmatrix} n-1\\ k-1\end{pmatrix}$个。

&emsp;&emsp;若有2个数落在了区间$[x,x+\delta x]$，同理有，

$$
P(E)=x^{k-2}(1-x-\delta x)^{n-k}(\delta x)^2=o(\delta x)
$$

因此，只要落入区间$[x,x+\delta x]$内的数字超过一个，则对应的概率为$o(\delta x)$。于是可以计算落入区间$[x,x+\delta x]$对应的概率，即

$$
\begin{split}
P(x\le X_{(k)}\le x+\delta x)&=n\begin{pmatrix} n-1\\ k-1\end{pmatrix}P(E)+o(\delta x)\\
&=n\begin{pmatrix} n-1\\ k-1\end{pmatrix} x^{k-1}(1-x)^{n-k}\delta x + o(\delta x)
\end{split}
$$

可以得到概率密度函数，

$$
\begin{split}
f(x)&=\lim_{\delta x\rightarrow 0}\frac{P(x\le X_{(k)}\le x+\delta x)}{\delta x}\\
&=n\begin{pmatrix} n-1\\ k-1\end{pmatrix} x^{k-1}(1-x)^{n-k}\\
&=\frac{n!}{(k-1)!(n-k)!}x^{k-1}(1-x)^{n-k}\\
&=\frac{\Gamma(n+1)}{\Gamma(k)\Gamma(n-k-1)}x^{k-1}(1-x)^{n-k}\\
\end{split}
$$

&emsp;&emsp;对上式，取$\alpha=k,\beta=n-k+1$，则可以得到**Beta分布的概率密度函数**，

$$
\mathrm{Beta}(x|\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}\tag{5}
$$

