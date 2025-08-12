# 共轭分布

## Gamma分布

### Gamma函数

&emsp;&emsp;**Gamma函数**是指形如以下的函数，

$$
\Gamma(x)=\int_0^{\infty} t^{x-1}e^{-t}dt
$$(gamma-pdf)

其导函数为，

$$
\frac{d\Gamma(x)}{dx}=\int_0^{\infty} \log(t)\times t^{x-1}e^{-t}dt
$$(gamma-derivative)

该函数有如下特点：

- **递推性**. $\Gamma(x)=(x-1)\Gamma(x-1)$; 易知，$\Gamma(x) $实际上是阶乘在实数集上的扩展，即$\Gamma(x)=(x-1)!, \forall x\in\mathbb{R}$.
- **特殊点**. $\Gamma'(1)=-\gamma$(欧拉常数)

|$x$ | $\Gamma(x)$ |
|:---:|:---:|
| $x=1$ | 1 |
| $x=\frac12$ | $\pi^{1/2}$ |



### Gamma分布

&emsp;&emsp;Gamma函数在统计中出现频率很高，比如常见的三大分布($t$分布，$\chi^2$分布，$F$分布)、Beta分布、Dirichlet分布等分布的概率密度函数中都有Gamma函数。最直接相关的则是Gamma分布，其概率密度函数为  

$$
\mathrm{Ga}(x|\alpha)=\frac{x^{\alpha-1}e^{-x}}{\Gamma(\alpha)}
$$(gamma-density)

由$\int_0^{\infty} \frac{t^{x-1}e^{-t}}{\Gamma(x)}dt=1$，易知上式即为Gamma分布的概率密度函数。

&emsp;&emsp;令$x=\beta t$，则可以得到更一般形式的Gamma分布，

$$
\mathrm{Ga}(t|\alpha,\beta)=\frac{\beta^\alpha t^{\alpha-1} e^{-\beta t}}{\Gamma(\alpha)}
$$(gamma-density-general)

其中，$\alpha$称为shape参数，主要决定了分布曲线的形状；$\beta$称为rate参数，主要决定曲线的陡峭。

&emsp;&emsp;**1. Gamma分布与Poisson分布**

&emsp;&emsp;参数为$\lambda$的Poissson分布，其分布律为，

$$
\mathrm{Poi}(X=k|\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}
$$(poisson-density)

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
\mathrm{Beta}(x|\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$(beta-density)

### Beta-Binomial共轭

&emsp;&emsp;如果继续均匀地在区间$[0,1]$随机抽取5个数，并告之每一个与数字$X_{(8)}$的大小，请问$X_{(8)}$这个数是多少？由于$p=X_{(k)}$这个数在$X_1,...,X_n,Y_1,...,Y_m$，$m+n$个数中中是第$k+m_1$大的，由Beta分布的概率密度函数可知，此时$p=X_{(k)}$的概率密度函数是$\mathrm{Beta}(p|k+m_1,n-k+1+m_2)$。

&emsp;&emsp;上述过程可以总结为：

1. $p=X_{(k)}$是要估计的参数，所推导出的$p$的分布为$f(p)=\mathrm{Beta}(p|k,n-k+1)$，称为$p$的先验分布；

2. 数据$Y_1,...,Y_m$中有$m_1$个数比$p$小，$Y_i$相当于做了$m$次伯努利实验，所以$m_1\sim B(m,p)$；

3. 在给定了来自数据提供的$(m_1,m_2)$的知识后，$p$的后验分布变为了$\mathrm{Beta}(p|k+m_1,n-k+1+m_2)$。

&emsp;&emsp;根据上述描述，可以将这一过程为，

$$
\textbf{先验分布}+\textbf{数据似然}=\textbf{后验分布}
$$

一般来说，对于非负实数$\alpha,\beta$，Beta分布有以下关系，

$$
\mathrm{Beta}(p|\alpha,\beta)+\textrm{BinomCount}(m_1,m_2)=\mathrm{Beta}(p|\alpha+m_1,\beta+m_2)
$$

&emsp;&emsp;该公式(6)实际上描述的就是Beta-Binomial共轭。**共轭**的意思就是：数据符合二项分布，参数的先验分布都能保持Beta分布的形式。好处是能够在先验分布中赋予参数明确的物理意义，该解释可以延续到后验分布。

### Dirichlet-Multinomial共轭

&emsp;&emsp;Beta分布拓展到多维，称为Dirichlet分布，即，

$$
\mathrm{Dir}(\pmb{x}|\pmb{\alpha})=\frac{\Gamma(\alpha_1+\alpha_2+...+\alpha_n)}{\Gamma(\alpha_1)\Gamma(\alpha_2)...\Gamma(\alpha_n)}x_1^{\alpha_1-1}x_2^{\alpha_2-1}...x_n^{\alpha_n-1}
$$(dirichlet-density)

与Beta分布类似，Dirichlet分布的共轭分布为Multinomial分布，

$$
\mathrm{Dir}(\pmb{p}|\pmb{\alpha})+\mathrm{MultCount}(\pmb{m})=\mathrm{Dir}(\pmb{p}|\pmb{\alpha}+\pmb{m}) 
$$

Multinomial分布为，

$$
\mathrm{Mult}(\pmb{n}|\pmb{p},N)=\begin{pmatrix} N\\ \pmb{n}\end{pmatrix}\prod_{k=1}^K p_k^{n_k}
$$(multinomial-density)

