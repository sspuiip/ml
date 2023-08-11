# 核函数基础
## 核函数

### 示例


&emsp;&emsp;首先来看一个核函数的例子。假设有XOR数据集$X = \{(0,0),(1,0),(0,1),(1,1)\}$，$y=\{0,1,1,0\}$，则该XOR异或问题在原输入空间很难分开两类样本，

![XOR输入空间](../img/inputspace.png)

&emsp;&emsp;如果我们对原输入空间样本$\pmb{x}$做如下映射$\phi :\mathbb{R}^2\rightarrow\mathbb{R}^3$，

$$
\phi: \pmb{x}=(x_1,x_2)\rightarrow \phi(\pmb{x})=(x_1^2,\sqrt{2}x_1x_2,x_2^2)
$$

不难发现，原空间不可分的XOR问题在新特征空间变的可分，如下图所示

![特征空间](../img/featurespace.png)

&emsp;&emsp;因此，原空间难以解决的问题映射至新特征空间后可以得到解决。特征映射成为了关键步骤。那么特征映射$\phi(\pmb{x})$是否必须显式的计算呢？下面，我们来分析一下新特征空间两个任意数据点$\phi(\pmb{x}),\phi(\pmb{z})$之间的相似性。采用未正则化的余弦相似性来度量，可知

$$
\begin{split}
\langle\phi(\pmb{x}),\phi(\pmb{z})\rangle&=\langle (x_1^2,\sqrt{2}x_1x_2,x_2^2),(z_1^2,\sqrt{2}z_1z_2,z_2^2)\rangle\\
&=x_1^2z_1^2+x_2^2z_2^2+2x_1x_2z_1z_2\\
&=(x_1z_1+x_2z_2)^2\\
&=\langle\pmb{x},\pmb{z}\rangle^2\\
&\triangleq\kappa(\pmb{x},\pmb{z})
\end{split}
$$

&emsp;&emsp;经过上式分析，我们发现特征映射后的向量做内积运算等价于原空间向量内积的平方。换句话说，我们可以不用显式计算映射$\phi$，直接计算$\kappa(\pmb{x},\pmb{z})$就可以实现特征映射后的效果，这一技巧也称之为核技巧(kernel trick), 函数$\kappa(\cdot,\cdot)$也称之为核函数。

![核函数示例](../img/kernelfun.jpeg)

---
### 什么是核函数

 >**定义1(内积(inner product))**. 令 $\mathcal{H}$ 为一个向量空间，则一个函数$\langle\cdot,\cdot\rangle_\mathcal{H}:\mathcal{H}\times\mathcal{H}\rightarrow\mathbb{R}$定义为$\mathcal{H}$的内积，如果满足以下条件：

 1. $\langle\alpha_1 f_1+\alpha_2 f_2,g\rangle_{\mathcal{H}}=\alpha_1\langle f_1,g\rangle_\mathcal{H}+\alpha_2\langle f_2,g\rangle_{\mathcal{H}}$
 2. $\langle f,g\rangle_{\mathcal{H}}=\langle g,f\rangle_{\mathcal{H}}$
 3. $\langle f,f\rangle_{\mathcal{H}}\ge 0 and \langle f,f\rangle_{\mathcal{H}}=0$ if and only if $f=0$

&emsp;&emsp;希尔伯空间是定义了内积的一个空间，并附加了一个技术性条件(A Hilber space is a space on which an inner product is defined, along with an additional technical condition)。



>**定义2(核函数, Kernel)**.  令$\mathcal{X}$为一个非空集合。一个函数 $k:\mathcal{X}\times\mathcal{X}\rightarrow \mathbb{R}$称之为核函数，如果存在一个$\mathbb{R}$-Hilbert空间以及映射 $\phi:\mathcal{X}\rightarrow \mathcal{H}$且满足$\forall x,x'\in \mathcal{X}$，

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle_\mathcal{H}
$$




>**Lemma(Sums of kernels are kernels)**. 给定$\alpha >0$以及 $k,k_1,k_2$为核函数定义在域$\mathcal{X}$, 则有$\alpha k$和$k_1+k_2$都是定义在$\mathcal{X}$的核函数.

>**Lemma(Mapping between spaces)**. 若$\mathcal{X}$ 和$\tilde{\mathcal{X}}$为非空集，且有一个映射$A:\mathcal{X}\rightarrow\tilde{\mathcal{X}}$. 若有$k$定义在域$\tilde{\mathcal{X}}$. 则$k(A(x),A(x'))$是一个定义在$\mathcal{X}$的核函数.

>**Lemma(Products of kernels are kernels)**. 给定$k_1$定义在域 $\mathcal{X}_1$ 以及$k_2$定义在域$\mathcal{X}_2$, 则 $k_1\times k_2$是一个定义在域$\mathcal{X}_1\times\mathcal{X}_2$的核函数。如果$\mathcal{X}_1=\mathcal{X}_2=\mathcal{X}$, 则 $k=k_1\times k_2$ 是一个定义在域$\mathcal{X}$的核函数。

---
#### 常用核函数

>**定义 (多项式核函数, Polynomial kernels)**. 假设$x,x'\in \mathbb{R}^d$ for $d\ge 1$，且$m\ge 1$是一个整数，以及$c\ge 0$是一个正实数，则有如下核函数：

$$
k(x,x')=(\langle x,x'\rangle+c)^m
$$


&emsp;&emsp;我们能否将求和与乘积法则的结合推广到有无限多项的求和?事实证明是可以的。

>**定义 ($l_p$空间)** 关于$p$-可求和序列的空间$\ell_p$，定义为对所有的$(a_i)_{i\ge 1}$都有

 $$
 \sum_{i=1}^{\infty}a_i^p <\infty
 $$

&emsp;&emsp;核函数可以用$\ell_2$序列来定义。

 >**Lemma**. 给定一个非空集合$\mathcal{X}$, 以及$\ell_2$空间的一个函数序列$(\phi_i(x))_{i\ge 1}$，其中映射$\phi_i :\mathcal{X}\rightarrow \mathbb{R}$ 是特征映射$\phi(x)$的第$i$th坐标，则有一个定义在域$\mathcal{X}$的核函数

 $$
 k(x,x')=\sum_{i=1}^\infty\phi_i(x)\phi_i(x')
 $$

&emsp;&emsp;泰勒级数展开可以用来定义具有无穷多个特征的核(Taylor series expansions may be used to define kernels that have infinityly many features)。

 >**定义 (泰勒级数核， Taylor series kernel)**. 假设我们可以定义泰勒级数

$$
f(z)=\sum_{n=0}^\infty a_n z^n\quad |z|<r,z\in\mathbb{R}
$$

for $r\in (0,\infty]$, with $a_n\ge 0$ for all $n\ge 0$. 可定义$\mathcal{X}$为空间$\mathbb{R}^d$的$\sqrt{r}$-球, 则如果$x,x'\in\mathbb{R}^d$满足$\Vert x\Vert <\sqrt{r}$, 我们有以下核函数

$$
k(x,x')=f(\langle x,x'\rangle)=\sum_{n=0}^\infty a_n\langle x,x'\rangle^n
$$

&emsp;&emsp;**证明**. 核的非负加权和是核，核的乘积是核，所以下面的是核，如果它收敛，

 $$
 k(x,x')=\sum_{n=1}^\infty a_n(\langle x,x'\rangle)^n
$$

根据Cauchy-Schwarz不等式

 $$
  |\langle x,x'\rangle |\le \Vert x\Vert\Vert x'\Vert<r
 $$

因此泰勒级数核收敛.

&emsp;&emsp;泰勒级数核的一个例子是指数核(An example of a Taylor series kernel is the exponential)。

>**定义 (指数核 (Exponential kernel))**. 域$\mathbb{R}^d$的指数核定义为 

 $$
 k(x,x')=\exp(\langle x,x'\rangle)
 $$

&emsp;&emsp;我们可以把以上所有的结果结合起来得到如下结果(乘积规则、映射规则等)



>**示例 (高斯核 Gaussian kernel)**. 定义在域$\mathbb{R}^d$的高斯核为 

 $$
 k(x,x')=\exp(-\gamma^{-2}\Vert x-x'\Vert^2)
 $$

 &emsp;&emsp;**证明**. 对于指数核函数$k_1$进行正则化， 

 $$
 \begin{split}
 k_1(x,x')&=\exp\left\{\frac{\langle x,x'\rangle}{\gamma^2}\right\}\\
\end{split}
 $$

 则有，

 $$
 \begin{split}
 \underbrace{k_1(x,x')}_{\textrm{normalize}}&=\frac{\exp\{\frac{\langle x,x'\rangle}{\gamma^2}\}}{  \sqrt{\exp\left\{\frac{\Vert x\Vert^2}{\gamma^2} \right\}    \exp\left\{\frac{\Vert x'\Vert^2}{\gamma^2} \right\}}     }\\
 &=\exp\left\{{\frac{2\langle x,x'\rangle}{\gamma^2}-\frac{\langle x,x\rangle}{\gamma^2}-\frac{2\langle x',x'\rangle}{\gamma^2}}\right\}\\
 &=\exp(-\gamma^{-2}\Vert x-x'\Vert^2)\\\\
 &=k(x,x')
 \end{split}
 $$

因此可得，高斯核为指数核$k_1$的标准化结果。显然高斯核也是一个核函数。

&emsp;&emsp;高斯核的特征映射为无穷维。下面对其特征映射进行分解，为简化计算，可先假设$\gamma =1 $。

$$
\begin{split}
\kappa(x,x')&=\exp(-x^2)\exp(-x'^2)\exp(2xx')\\
&=\exp(-x^2)\exp(-x'^2)\underbrace{\sum_{n=0}^\infty\frac{(2xx')^n}{n!}}_{\textrm{Taylor}展开式}\\
&=\sum_{n=0}^\infty \exp(-x^2)\exp(-x'^2)\sqrt{\frac{2^n}{n!}}\sqrt{\frac{2^n}{n!}}x^n x'^n\\
&=\sum_{n=0}^\infty\underbrace{\left(\exp(-x^2)\sqrt{\frac{2^n}{n!}}x^n\right)}_{\phi(x)}\underbrace{\left(\exp(-x'^2)\sqrt{\frac{2^n}{n!}}x'^n\right)}_{\phi(x')}
\end{split}
$$

可知，特征映射为，

$$
\phi(x)=\exp(-x^2)(1,\sqrt{\frac{2^1}{1!}}x^1,\sqrt{\frac{2^2}{2!}}x^2,...,\sqrt{\frac{2^n}{n!}}x^n,...)
$$

---
## 特征映射的基本运算

1. 模长

$$
\Vert \phi(\pmb{x})\Vert=\sqrt{\langle \phi(\pmb{x}),\phi(\pmb{x})\rangle}=\sqrt{\kappa(\pmb{x},\pmb{x})}
$$

2. 标准化

$$
\hat{\phi}(\pmb{x})=\frac{\phi(\pmb{x})}{\Vert\phi(\pmb{x}) \Vert}
$$

$$
\hat{\kappa}(\pmb{x},\pmb{z})=\frac{\kappa(\pmb{x},\pmb{z})}{\sqrt{\kappa(\pmb{x},\pmb{x})\kappa(\pmb{z},\pmb{z})}}
$$

3. 线性组合

$$
\left\Vert \sum_{i=1}^la_i\phi(\pmb{x}_i)\right\Vert^2=\sum_{i,j=1}^l a_ia_j\kappa(\pmb{x}_i\pmb{x}_j)
$$

4. 距离

$$
\left\Vert \phi(\pmb{x})-\phi(\pmb{z})\right\Vert^2=\kappa(\pmb{x},\pmb{x})-2\kappa(\pmb{x},\pmb{z})+\kappa(\pmb{z},\pmb{z})
$$

5. 均值

$$
\left\Vert \phi_S \right\Vert^2=\left\Vert \frac{1}{l}\sum_{i=1}^l\phi(\pmb{x}) \right\Vert^2=\frac1l\sum_{i=1}^l\sum_{j=1}^l\kappa(\pmb{x}_i,\pmb{x}_j)
$$

6. 样本均值距离

$$
\left\Vert \phi(\pmb{x})-\phi_S\right\Vert^2=\kappa(\pmb{x},\pmb{x})-\frac2l\sum_{i=1}^l\kappa(\pmb{x},\pmb{x}_i)+\frac{1}{l^2}\sum_{i=1}^l\sum_{j=1}^l\kappa(\pmb{x}_i,\pmb{x}_j)
$$

7. 平均样本均值距离

$$
\frac1l\sum_{i=1}^l\left\Vert \phi(\pmb{x}_i)-\phi_S\right\Vert^2=\frac1l\sum_{i=1}\kappa(\pmb{x}_i,\pmb{x}_j)-\frac{1}{l^2}\sum_{i=1}\sum_{j=1}\kappa(\pmb{x}_i,\pmb{x}_j)
$$

8. 中心化

- 样本中心化


$$
\tilde{\phi}(\pmb{x})=\phi(\pmb{x})-\phi_S
$$

- 核函数中心化

$$
\begin{split}
\tilde{\kappa}(\pmb{x},\pmb{z})&=\langle\phi(\pmb{x})-\phi_S,\phi(\pmb{x})-\phi_S\rangle\\
&=\kappa(\pmb{x},\pmb{z})-\frac1l\sum_{i=1}\kappa(\pmb{x}_i,\pmb{z})-\frac{1}{l}\kappa\sum_{i=1}\kappa(\pmb{x}_i,\pmb{x})+\frac{1}{l^2}\sum_{i,j=1}\kappa(\pmb{x}_i,\pmb{x}_j)\\
\tilde{\pmb{K}}&=\pmb{K}-\frac1l\pmb{K11}^\top-\frac1l\pmb{11}^\top\pmb{K}+\frac{1}{l^2}(\pmb{1}^\top\pmb{K}\pmb{1})\pmb{11}^\top
\end{split}
$$

---
## 投影

 $\phi(\pmb{x})$在向量$\pmb{w}$上的**投影**$P_{\pmb{w}}(\phi(\pmb{x}))$为，

$$
P_{\pmb{w}}(\phi(\pmb{x}))=\frac{\langle\phi(\pmb{x}),\pmb{w}\rangle}{\Vert\pmb{w}\Vert}\cdot\frac{\pmb{w}}{\Vert\pmb{w}\Vert}=\frac{\langle\phi(\pmb{x}),\pmb{w}\rangle}{\Vert\pmb{w}\Vert^2}\cdot\pmb{w}
$$

如果$\pmb{w}$已单位化，则有，

$$
P_{\pmb{w}}(\phi(\pmb{x}))=\langle\phi(\pmb{x}),\pmb{w}\rangle\cdot\pmb{w}=\pmb{w}\cdot\langle\phi(\pmb{x}),\pmb{w}\rangle=\pmb{w}\pmb{w}^\top\phi(\pmb{x})
$$

即，**正交投影**$ P_{\pmb{w}}^\bot \phi(\pmb{x}) $ 为，

$$
P_{\pmb{w}}^\bot\phi(\pmb{x})=(\pmb{I}-\pmb{ww}^\top)\phi(\pmb{x})
$$
