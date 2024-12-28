# 计算学习理论

&emsp;&emsp;计算学习理论是指通过计算来学习的理论[^mlzhou]。

&emsp;&emsp;假设有样本集$D=\{(\pmb{x}_1,y_1),(\pmb{x}_2,y_2),...,(\pmb{x}_m,y_m)\},\pmb{x}_i\in\mathcal{X}, y_i\in\mathcal{Y}$，所有样本服从一个隐含未知的分布$\mathcal{D}$，$D$中所有样本都是独立地从这个分布上采样而来。

&emsp;&emsp;令$h:\mathcal{X}\rightarrow \mathcal{Y}$，则其**泛化误差**为，

$$
E[h;\mathcal{D}]=P_{\pmb{x}\in\mathcal{D}}[h(\pmb{x})\neq y]
$$(gen-error)

$h$在$D$上的**经验误差**为，

$$
\hat{E}[h;D]=\frac{1}{m}\sum_{i=1}^m\mathbb{I}[h(\pmb{x}_i)\neq y_i]
$$(exp-error)

由于$D$是$\mathcal{D}$的独立同分布采样，因此$h$的经验误差的期望等于其泛化误差。若$h$在数据集$D$上的经验误差为0，则称$h$与$D$**一致**，否则称其与$D$**不一致**。任意两个映射$h_1,h_2\in \mathcal{X}\rightarrow\mathcal{Y}$，可以通过其**不合**来度量它们之间的差别，

$$
d(h_1,h_2)=P_{\pmb{x}\in\mathcal{D}}[h_1(\pmb{x})\neq h_2(\pmb{x})]
$$(eq-consistant)


1. **常用不等式**

   - Jensen不等式。对于任意凸函数$f(x)$有，

   $$
   f(\mathbb{E}[x])\le\mathbb{E}[f(x)]
   $$(eq-jensen)

   - Hoeffding不等式。若$x_1,...,x_m$为$m$个独立随机变量，且满足$0\le x_i\le 1$，则对于任意$\epsilon>0$有，

   $$
   P\left(\frac1m\sum_{i=1}^m x_i-\frac1m\sum_{i=1}^m\mathbb{E}(x_i)\ge\epsilon \right)\le\exp(-2m\epsilon^2)
   $$(eq-hoeffding)
   
   - McDiamid不等式。若$x_1,...,x_m$为$m$个独立随机变量，且对于任意$1\le i\le m$，函数$f$满足，

     $$
     \sup_{x_1,...,x_m,x_i'}\left| f(x_1,...,x_m)-f(x_1,...,x_{i-1},x_i',x_{i+1},...,x_m) \right|\le c_i
     $$

     则对于任意$\epsilon>0$，有

     $$
     P\left( f(x_1,...,x_m) -\mathbb{E}[f(x_1,...,x_m)] \ge\epsilon\right)\le \exp\left(\frac{-2\epsilon^2}{\sum_i c_i^2}\right)
     $$(eq-mcdiamid)
     

## 概率近似正确学习理论

&emsp;&emsp;计算学习理论中最基本的是概率近似正确(probably approximately correct, PAC)学习理论。

&emsp;&emsp;令$c$表示**概念**，即从样本空间$\mathcal{X}$到标记空间$\mathcal{Y}$的映射。若对于任意样本$(\pmb{x}_i,y_i),i\in\{1,2,...\}$，有$c(\pmb{x}_i)=y_i$，则称$c$为**目标概念**。所有希望学得的目标概念所构成的集合称为**概念类**(concept class)，记为$\mathcal{C}$。

&emsp;&emsp;给定学习算法$\mathfrak{L}$，它所考虑的所有可能概念的集合称为**假设空间**（hypothesis space）,记为$\mathcal{H}$。由于学习算法并不知道概念类的真实存在，因此，$\mathcal{H}$和$\mathcal{C}$通常是不同的，学习算法会把自认为可能的目标概念集中起来构成$\mathcal{H}$。对于$h\in\mathcal{H}$，由于并不能确定它是否为真目标概念，因此称为**假设**。

&emsp;&emsp;若目标概念$c\in\mathcal{H}$，则$\mathcal{H}$中存在假设能将所有示例按与真实标记一致的方式完全分开，则称该问题对学习算法$\mathfrak{L}$是**可分的**(separable)，亦称为**一致的**(consistent)。反之，若$c\notin\mathcal{H}$，则称之为不可分的、不一致的。

&emsp;&emsp;给定训练集$D$，我们希望基于学习算法$\mathfrak{L}$所学模型的对应假设$h$尽可能接近目标概念$c$。但由于机器学习过程受到多种因素制约，例如:

- 训练集样本数有限

&emsp;&emsp;由于样本数有限，通常会存在一些在$D$上“等效”的假设，学习算法对它们无法区别。

- 采样得到训练集$D$的偶然性

&emsp;&emsp;即便是同样大小的训练集，学得结果也可能有所不同。

&emsp;&emsp;因此，我们还是希望以较大的把握学得比较好的模型。也就是说，以较大的概率学得误差满足预设上限的模型。这就是“概率”、“近似正确”的含义。

### PAC

&emsp;&emsp;若定义$\delta$为置信度，则形式上可以给出PAC辩识的定义如下，

>**定义** (**PAC辩识**). 对于$0<\epsilon, \delta<1$，所有$c\in\mathcal{C}$和分布$\mathcal{D}$，若存在学习算法$\mathfrak{L}$，其输出假设$h\in\mathcal{H}$满足

$$
P(E(h)\le\epsilon)\ge 1-\delta
$$(def-pac-identify)

则称学习算法$\mathfrak{L}$能从假设空间$\mathcal{H}$中PAC辩识概念类$\mathcal{C}$。

>**定义** (**PAC可学习**). 令$m$表示从分布$\mathcal{D}$中独立同分布采样得到的样例数目，$0<\epsilon, \delta<1$，对于所有分布$\mathcal{D}$，若存在学习算法$\mathfrak{L}$和多项式函数$\textrm{ploy}()$，使得对于**任何$m\ge \textrm{ploy}(1/\epsilon,1/\delta,size(x),size(c))$**，$\mathfrak{L}$能从假设空间$\mathcal{H}$中PAC辩识概念类$\mathcal{C}$，则称概念类$\mathcal{C}$对于假设空间$\mathcal{H}$而言是PAC可学习的。也称之为概念类$\mathcal{C}$是PAC可学习的。

>**定义** (**PAC学习算法**). 若学习算法$\mathfrak{L}$使概念类$\mathcal{C}$为PAC可学习的，且$\mathfrak{L}$的**运行时间**也是多项式函数$\textrm{ploy}(1/\epsilon,1/\delta,size(x),size(c))$，则称概念类$\mathcal{C}$是高效PAC可学习的，称$\mathfrak{L}$为概念类$\mathcal{C}$的PAC学习算法。

>**定义** (**样本复杂度**). 满足PAC学习算法$\mathfrak{L}$所需$m\ge \textrm{ploy}(1/\epsilon,1/\delta,size(x),size(c))$的**最小值**，称为学习算法$\mathfrak{L}$的样本复杂度。



> **显然，PAC学习给出了一个抽象刻画机器学习能力的框架。基于这个框架，能对很多问题进行探讨：例如研究某任务在什么条件下可学习到较好模型？需要多少样本才能获得好模型？某算法在什么条件下可进行有效学习等。**



#### 有限假设空间($|\mathcal{H}|$有限)

1. **可分情况**($c\in\mathcal{H}$)

&emsp;&emsp;（1）模型。该情况下，使用“排除法”，给定$m$个样本集$D$可以找到满足误差参数$\epsilon$的假设$h$。首先排除与$D$不一致的假设，直到$\mathcal{H}$中只剩下1个为止。假设空间可能存在不止一个与$D$一致的等效假设，对这些假设，无法根据$D$来对它们的优劣进一步区分。

&emsp;&emsp;（2）样本数。到底需要多少样本才能学习到目标概念$c$的有效近似呢？对PAC来说，只要训练集$D$的规模能让学习算法$\mathfrak{L}$以概率$1-\delta$找到目标假设的$\epsilon$近似即可。首先估算泛化误差大于$\epsilon$但在数据集上仍表现完美的假设出现的概率，即

$$
\begin{split}
P[h(\pmb{x})=y]&=1-P[h(\pmb{x})\neq y]\\
&=1-E[h] \\
&\le 1-\epsilon
\end{split}
$$

&emsp;&emsp;由于$D$包含了$m$个从$\mathcal{D}$独立同分布采样而得的样例，因此，$h$与$D$表现一致的概率为，

$$
\begin{split}
P[h(\pmb{x}_1=y_1)\wedge h(\pmb{x}_2=y_2)\wedge ...\wedge h(\pmb{x}_m=y_m)]&=(1-P[h(\pmb{x})\neq y])^m\\
&\le (1-\epsilon)^m
\end{split}
$$

&emsp;&emsp;由于事先并不知道学习算法$\mathfrak{L}$输出$\mathcal{H}$中的哪个假设，但仅需保证泛化误差大于$\epsilon$，且在$D$表现完美的所有假设出现概率之和不大于$\delta$即可：

$$
\begin{split}
P(h\in\mathcal{H}:E[h]>\epsilon\wedge\hat{E}[h]=0)&<|\mathcal{H}|(1-\epsilon)^m\\
&<\underbrace{|\mathcal{H}|e^{-m\epsilon}}_{e^{-x}>1-x}
\end{split}
$$

只要令上式不大于$\delta$，即，

$$
|\mathcal{H}|e^{-m\epsilon}\le \delta
$$

则可以得到，

$$
m\ge \frac{1}{\epsilon}\left(\ln|\mathcal{H}|+\ln\frac{1}{\epsilon} \right)
$$

&emsp;&emsp;由此可知，有限假设空间$\mathcal{H}$都是PAC可学习的，所需的样本数如上式所示，输出假设$h$的泛化误差随着样例数目的增多而收敛到0，收敛速率为$O(1/m)$。

2. **不可分情况**($c\notin\mathcal{H}$)

&emsp;&emsp;（1）模型。目标概念$c$往往不存在于假设空间$\mathcal{H}$，假定对于任何$h\in\mathcal{H},\hat{E}(h)\neq 0$，也就是$\mathcal{H}$中的任意假设都会在训练集上出现或多或少的错误。由Hoeffding不等式可知，

>**引理** 1.若训练集$D$包含$m$个从分布$\mathcal{D}$上独立同分布采样而得到的样例, $0<\epsilon<1$，则对于任意$h\in \mathcal{H}$有，

$$
\begin{split}
P(\hat{E}(h)-E(h)\ge\epsilon)&\le\exp(-2m\epsilon^2)\\
P(E(h)-\hat{E}(h)\ge\epsilon)&\le\exp(-2m\epsilon^2)\\
P(|\hat{E}(h)-E(h)|\ge\epsilon)&\le 2\exp(-2m\epsilon^2)\\
\end{split}
$$(eq-not-split)

>**推论** 若训练集$D$包含$m$个从分布$D$上独立同分布采样而得到样本， $0<\epsilon<1$，则对任意$h\in\mathcal{H}$，下式至少以$1-\delta$的概率成立，

$$
\hat{E}(h)-\sqrt{\frac{\ln(2/\delta)}{2m}}\le E(h)\le \hat{E}(h)+\sqrt{\frac{\ln(2/\delta)}{2m}}
$$(eq-hyposis-error)

上式说明，样本数$m$较大时，$h$的经验误差是其泛化误差较好的近似。

>**定理** 1 若$\mathcal{H}$为有限假设空间，$0<\delta<1$，则对任意$h\in\mathcal{H}$有，

$$
P\left(|E(h)-\hat{E}(h)|\le\sqrt{\frac{\ln|\mathcal{H}|+\ln(2/\delta)}{2m}} \right)\ge1-\delta
$$(thm-hyposis)

&emsp;&emsp;显然，当$c\notin\mathcal{H}$时，学习算法$\mathfrak{L}$无法学得目标概念$c$的近似。但是，当$\mathcal{H}$给定时，必存在一个泛化误差最小的假设，找出此假设的$\epsilon$近似也不失为一个较好的目标。$\mathcal{H}$中泛化最小的假设是$\arg\min\limits_{h\in\mathcal{H}}E(h)$，于是，以此为目标可将PAC学习推广至$c\notin\mathcal{H}$的情况，这也称之为“不可知学习”(agnostic learning)，即

>**定义** (**不可知PAC学习**). 若存在学习算法$\mathfrak{L}$和多项式函数$\text{ploy}()$，使得对于任意$m\ge\text{ploy}(1/\epsilon,1/\delta,size(\pmb{x}),size(c))$，$\mathfrak{L}$都能从假设空间$\mathcal{H}$输出满足下式的假设$h$:

$$
P\{E(h)-\arg\min\limits_{h'\in\mathcal{H}}E(h')\le\epsilon\}\ge 1-\delta
$$(def-no-pac-learning)

则称假设空间$\mathcal{H}$是不可知PAC可学习的。


### VC维
#### 无限假设空间($|\mathcal{H}|$无限)

&emsp;&emsp;对于无限假设空间的学习，需要度量*假设空间的复杂度*。最常见的办法是考虑假设空间的“VC维”。

>**定义** (**VC维**). VC维就是假设空间$\mathcal{H}$能打散的最大示例集大小，即

$$
\text{VC}(\mathcal{H})=\max\limits_{m}\{m: \Pi_{\mathcal{H}}(m)=2^m\}
$$(eq-vc)

&emsp;&emsp;所谓的**打散**是指：如果假设空间$\mathcal{H}$能对示例集$D$实现*所有不相同的对分*，则称为$D$能被$\mathcal{H}$打散。而**对分**(赋予标记一次)是指假设空间$\mathcal{H}$的假设$h$对示例集$D$中示例赋予的一种结果。$\Pi_{\mathcal{H}}(m)$为**增长函数**，即

>**定义** (**增长函数**). $\forall m\in\mathbb{N}$，设空间$\mathcal{H}$的增长函数$\Pi_{\mathcal{H}}(m)$为，
```{math}
:label: eq-inc-fun
\Pi_{\mathcal{H}}(m)=\max\limits_{\{\pmb{x}_1,...,\pmb{x}_m\}\subseteq\mathcal{X}}\left|\{(h(\pmb{x}_1),...,h(\pmb{x}_m)|h\in\mathcal{H}\} \right|
```

&emsp;&emsp;增长函数表示假设空间$\mathcal{H}$对$m$个示例所能赋予标记的最大可能数目。这一结果越大，则$\mathcal{H}$的表示能力越强，对学习任务的适应能力越强。我们可以利用增长函数来估计经验误差与泛化误差之间的关系。

>**定理** 2. 对假设空间$\mathcal{H}$， $m\in\mathbb{N}, 0<\epsilon<1$和任意$h\in\mathcal{H}$有，
```{math}
:label: inc-fun-error
P(|E(h)-\hat{E}(h)|>\epsilon)\le 4\Pi_{\mathcal{H}}(2m)\exp\left(-\frac{m\epsilon^2}{8} \right)
```



&emsp;&emsp;(一) **VC维计算**

{attribution="VC维计算方法"}
>若存在大小为$d$的示例数据集能被$\mathcal{H}$打散，但不存在任何大小为$d+1$的示例集被$\mathcal{H}$打散，则$\mathcal{H}$的VC维是$d$。

&emsp;&emsp;**例1**. 对于实数域区间$[a,b]$，令$\mathcal{H}$表示实数域中所有闭区间构成的集合$\{h_{[a,b]}:a,b\in\mathbb{R},a\le b\}$, $\mathcal{X}=\mathbb{R}$。对于$x\in\mathcal{X}$，若$x\in [a,b]$，则$h_{[a,b]}=+1$，否则$h_{[a,b]}=-1$。若有$x_1=0.5, x_2=1.5$，则假设空间$\{h_{[0,1]},h_{[0,2]},h_{[1,2]},h_{[2,3]}\}$将$x_1,x_2$打散，所以$\mathcal{H}$的VC维至少为2；对于任意大小为3的示例集$\{x_1,x_2,x_3\}$，不妨设$x_1<x_2<x_3$，则$\mathcal{H}$中不存在任何假设$h_{[a,b]}$能实现对分结果$\{(x_1,+),(x_2,-),(x_3,+)\}$。因此$\mathcal{H}$的VC维为2。

&emsp;&emsp;(二) **VC维与增长函数关系**

&emsp;&emsp;由VC维的定义可知，VC维与增长函数有着密切联系。以下定理给出了二者之间的定量关系。

>**定理 (Sauer引理)**. 若假设空间$\mathcal{H}$的VC维为$d$，则对任意$m\in\mathbb{N}$有，
```{math}
:label: eq-sauer
\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^d\begin{pmatrix}m\\ i \end{pmatrix}
```
证明略。由Sauer引理，可计算出增长函数的上界：

>**推论**. 若假设空间$\mathcal{H}$的VC维为$d$，则对任意整数$m\ge d$有，
```{math}
:label: sauer-infer
\Pi_{\mathcal{H}}(m)\le \left(\frac{e\cdot m}{d}\right)^d
```

&emsp;&emsp;(三) **VC维与泛化误差界**
>**定理 3**. 若假设空间$\mathcal{H}$的VC维为$d$，则对任意$m>d,0<\delta<1$和$h\in\mathcal{H}$有
```{math}
:label: vc-error-relation
\boxed{
P\left(\left| E(h)-\hat{E}(h) \right|\le\sqrt{\frac{8d\ln\frac{2em}{d}+8\ln\frac{4}{\delta}} {m}   } \right) \ge 1-\delta}
```
证明. 令$\delta=4\Pi_{\mathcal{H}}(2m)\exp\left(-\frac{m\epsilon^2}{8}\right)\le 4 \left(\frac{e\cdot m}{d}\right)^d\exp\left(-\frac{m\epsilon^2}{8}\right)$，可解得$\epsilon = \sqrt{\frac{8d\ln\frac{2em}{d}+8\ln\frac{4}{\delta}} {m}   }$。

&emsp;&emsp;由此可知，泛化误差界{eq}`vc-error-relation`只与样例数$m$有关，收敛速率为$O(\sqrt{\frac{1}{m}})$，与数据分布$\mathcal{D}$和样例集$D$无关。因此，基于VC维的泛化误差界是分布无关、数据独立的。

>**定义 (经验风险最小化)**. 令$h$表示学习算法$\mathfrak{L}$输出的假设，若$h$满足
```{math}
:label: eq-exper-min
\hat{E}(h)=\min\limits_{h'\in\mathcal{H}}\hat{E}(h')
```
则称$\mathfrak{L}$为满足经验风险最小化原则的算法。


&emsp;&emsp;(四) **VC维与PAC可学习**

>**定理**. 任何VC维有限的假设空间$\mathcal{H}$都是（不可知）PAC可学习的。

- **样本量上界与VC维[^myref]**

&emsp;&emsp;如果$1\le d <\infty$，那么学习$\epsilon,\delta$所需的样本量为，
```{math}
:label: sample-count-up
m\le\frac{64}{\epsilon^2}\left[2d\ln\left(\frac{12}{\epsilon} \right) +\ln\left(\frac{4}{\delta} \right) \right]
```

- **样本量下界与VC维[^myref]**

&emsp;&emsp;如果$1\le d <\infty$，那么学习$\epsilon>0,\delta<\frac{1}{64}$所需的样本量至少为$\frac{d}{320\epsilon^2}$，此外，如果$\mathcal{H}$中至少包含2个$h$，那么对于$0<\epsilon<1$和$0<\delta<\frac14$来说，样本量至少为
```{math}
:label: sample-count-up
m\ge 2\left[ \frac{1-\epsilon^2}{2\epsilon^2}\ln \left(\frac{1}{8\delta(1-2\delta)} \right) \right]
```


### Rademacher复杂度

&emsp;&emsp;基于VC维的泛化误差界是分布无关数据独立的，也就是说对任意数据分布都成立，使得基于VC维的可学习性分析结果具有一定的普适性；但从另一方面来说，由于没有考虑数据本身，基于VC维得到的泛化误差界比较松。Rademacher复杂度是另一种描述假设空间复杂度的途径，与VC维不同，它在一定程度上考虑了数据分布。

&emsp;&emsp;假设$h$的经验误差为，

$$
\begin{split}
\hat{E}(h)&=\frac1m\sum_{i=1}^m \mathbb{I}(h(\pmb{x}_i)\neq y_i)\\
&=\frac1m\sum_{i=1}^m\frac{1-y_ih(\pmb{x}_i)}{2}\\
&=\frac12-\frac{1}{2m}\sum_{i=1}^m y_ih(\pmb{x}_i)
\end{split}
$$(hyposis-experi)

其中，$\frac{1}{m}\sum_{i=1}^m y_ih(\pmb{x}_i)$体现了预测值$h(\pmb{x}_i)$与样例真实标记$y_i$之间的一致性，若对于所有$i$都有$h(\pmb{x}_i)=y_i$，则$\frac{1}{m}\sum_{i=1}^m y_ih(\pmb{x}_i)$取最大值1。换句话说，经验误差最小的假设满足下式，

$$
\arg\max\limits_{h\in\mathcal{H}}\frac1m\sum_{i=1}^m y_ih(\pmb{x}_i)
$$(exp-error-min)

&emsp;&emsp;然而，现实应用中，样本标记可能会受到噪声影响，也就是某些标记$y_i$可能受随机因素影响不再是$\pmb{x}_i$的真实标记。**这种情况下，选择假设空间$\mathcal{H}$在训练集上表现良好的假设，有时还不如选择$\mathcal{H}$中事先已考虑了随机噪声影响的假设**。

&emsp;&emsp;考虑随机变量$\sigma_i$，且$P(\sigma_i=+1)=0.5=P(\sigma_i=-1)$，称为**Rademacher随机变量**，将此变量引入{eq}`exp-error-min`，则有，

$$
\sup\limits_{h\in\mathcal{H}}\frac1m\sum_{i=1}^m\sigma_i h(\pmb{x}_i)
$$(radermacher-error)

考虑$\mathcal{H}$中所有假设，可对式{eq}`radermacher-error`取期望，则有
```{math}
:label: all-rade-expectation
\mathbb{E}_{\pmb{\sigma}}\left[\sup\limits_{h\in\mathcal{H}}\frac1m\sum_{i=1}^m\sigma_i h(\pmb{x}_i) \right]
```
其中，$\pmb{\sigma}=\{\sigma_1,...,\sigma_m\}$。式{eq}`all-rade-expectation`的取值范围是$[0,1]$，它体现了假设空间$\mathcal{H}$的表示能力。例如：当$|\mathcal{H}|=1$时，只有一个假设，这时可计算出式{eq}`all-rade-expectation`的值为0；当$|\mathcal{H}|=2^m$且$\mathcal{H}$能打散$D$时，任意$\pmb{\sigma}$中总有一个假设使得$h(\pmb{x}_i)=\sigma_i (i=1,2,3...,m)$，此时式{eq}`all-rade-expectation`的值为1。

>**定义 (经验Rademacher复杂度)**. 函数空间$\mathcal{F}$关于样例集$Z$的经验Rademacher复杂度为，
```{math}
:label: f-z-rade-comp
\hat{R}_Z (\mathcal{F})=\mathbb{E}_{\pmb{\sigma}}\left[\sup\limits_{f\in\mathcal{F}}\frac{1}{m} \sum_{i=1}^m\sigma_i f(\pmb{z}_i) \right]
```

>**定义 (Rademacher复杂度)**. 函数空间$\mathcal{F}$关于$\mathcal{Z}$上分布$\mathcal{Z}$的Rademacher复杂度为，
```{math}
:label: f-z-rade-comp
R_m (\mathcal{F})=\mathbb{E}_{Z\subseteq\mathcal{Z}:|Z|=m }\left[\hat{R}_Z (\mathcal{F}) \right]
```

&emsp;&emsp;基于Rademacher复杂度可以得到关于函数空间$\mathcal{F}$的泛化误差界。

>**定理**. 对实值函数空间$\mathcal{F}:\mathcal{Z}\rightarrow [0,1]$，根据分布$\mathcal{D}$从$\mathcal{Z}$中独立同分布采样得到示例集$Z=\{\pmb{z}_1,...,\pmb{z}_m\}, \pmb{z}_i\in\mathcal{Z}$，$0<\delta<1$，对任意$f\in\mathcal{F}$，以至少$1-\delta$的概率有，
```{math}
:label: rad-error-bound
\boxed{
\begin{split}
\mathbb{E}[f(\pmb{z})]&\le \frac1m\sum_{i=1}^m f(\pmb{z}_i) + 2R_m(\mathcal{F})+\sqrt{\frac{\ln(1/\delta)}{2m}}\\
\mathbb{E}[f(\pmb{z})]&\le \frac1m\sum_{i=1}^m f(\pmb{z}_i) + 2\hat{R}_Z(\mathcal{F})+3\sqrt{\frac{\ln(2/\delta)}{2m}}\\
\end{split}}
```

>**定理**. 对假设空间$\mathcal{H}:\mathcal{X}\rightarrow\{-1,+1\}$，根据分布$\mathcal{D}$从$\mathcal{X}$中独立同分布采样得到示例集$D=\{\pmb{x}_1,...,\pmb{x}_m\}, \pmb{x}_i\in\mathcal{X}$，$0<\delta<1$，对任意$h\in\mathcal{H}$，以至少$1-\delta$的概率有，
```{math}
:label: rad-error-bound
\boxed{
\begin{split}
\mathbb{E}[h]&\le \hat{E}(h)+ R_m(\mathcal{H})+\sqrt{\frac{\ln(1/\delta)}{2m}}\\
\mathbb{E}[h]&\le \hat{E}(h)+ \hat{R}_D(\mathcal{H})+3\sqrt{\frac{\ln(2/\delta)}{2m}}\\
\end{split}}
```
>**定理**.  假设空间$\mathcal{H}$的Rademacher复杂度$R_m(\mathcal{H})$与增长函数$\Pi_\mathcal{H}(m)$满足，
```{math}
:label: rade-increase
R_m(\mathcal{H})\le \sqrt{\frac{2\ln\Pi_{\mathcal{H}}(m)}{m}}
```

&emsp;&emsp;由此可知，从Rademacher复杂度和增长函数能推导出基于VC维的泛化误差界，

$$
\boxed{
  E(h)\le\hat{E}(h)+\sqrt{\frac{2d\ln\frac{em}{d}}{m}}+\sqrt{\frac{\ln(1/\delta)}{2m}}
}
$$(conclusion)


可以发现，无论是基于VC维还是Rademacher复杂度来推导泛化误差界，所得到的结果均与具体学习算法无关，对所有学习算法都适用。这使得人们能够脱离具体学习算法的设计来考虑学习问题本身的性质。

### 稳定性

&emsp;&emsp;稳定性主要考查算法在输入变化时，输出是否随之发生较大变化。

>**定义** ($\beta$-均匀稳定性).  对任意$\pmb{x}\in\mathcal{X}, z=(\pmb{x},y)$，若学习算法$\mathfrak{L}$满足，
```{math}
:label: beta-stability
|\ell(\mathfrak{L}_D,z) - \ell(\mathfrak{L}_{D^{\backslash i}}, z) |\le \beta
```
则称算法$\mathfrak{L}$关于损失$\ell$满足$\beta$-均匀稳定性。


>**定理** (稳定性泛化误差界)  若算法$\mathfrak{L}$关于损失$\ell$满足$\beta$-均匀稳定性，且损失函数$\ell$的上界为$M$， $0<\delta<1$，则对于任意$m\ge 1$（$m$为从分布$\mathcal{D}$上独立同分步采样得到的数据集$D$的大小），至少以$1-\delta$的概率有，
```{math}
:label: stability-error-bound
\ell(\mathfrak{L},D)\le\hat{\ell}(\mathfrak{L},D)+2\beta+(4m\beta+M)\sqrt{\frac{\ln(1/\delta)}{2m}}
```

&emsp;&emsp;上述定理给出了基于稳定性分析推导出的学习算法$\mathfrak{L}$学习得到的假设的泛化误差界。稳定性分析不必考虑假设空间所有可能假设，只需要根据算法自身的稳定性来讨论输出假设$\mathfrak{L}_D$的泛化误差界。



[^mlzhou]: 周志华. 机器学习. 北京：清华大学出版社，2016.
[^myref]: Sanjeev Kulkarni, Gilbert Harman et al. An elementary introduction to statistical learning theory.