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

### 什么是核函数

 **定义1(inner product)**. Let $\mathcal{H}$  be a vector space over $\mathbb{R}$. A function $\langle\cdot,\cdot\rangle_\mathcal{H}:\mathcal{H}\times\mathcal{H}\rightarrow\mathbb{R}$ is said to be an inner product on $\mathcal{H}$ if

 1. $\langle\alpha_1 f_1+\alpha_2 f_2,g\rangle_{\mathcal{H}}=\alpha_1\langle f_1,g\rangle_\mathcal{H}+\alpha_2\langle f_2,g\rangle_{\mathcal{H}}$
 2. $\langle f,g\rangle_{\mathcal{H}}=\langle g,f\rangle_{\mathcal{H}}$
 3. $\langle f,f\rangle_{\mathcal{H}}\ge 0 and \langle f,f\rangle_{\mathcal{H}}=0$ if and only if $f=0$

&emsp;&emsp;A Hilber space is a space on which an inner product is defined, along with an additional technical condition.



**定义2(Kernel)**.  Let $\mathcal{X}$  be a non-empty set. A fucntion $k:\mathcal{X}\times\mathcal{X}\rightarrow \mathbb{R}$ is called a kernel if there exists an $\mathbb{R}$-Hilbert space and a map $\phi:\mathcal{X}\rightarrow \mathcal{H}$ such that $\forall x,x'\in \mathcal{X}$，

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle_\mathcal{H}
$$




**Lemma(Sums of kernels are kernels)**. Given $\alpha >0$ and $k,k_1,k_2$ all kernels on $\mathcal{X}$, then $\alpha k$ and $k_1+k_2$ are kernels on $\mathcal{X}$.

**Lemma(Mapping between spaces)**. Let $\mathcal{X}$ and $\tilde{\mathcal{X}}$ be sets, and define a map $A:\mathcal{X}\rightarrow\tilde{\mathcal{X}}$. Define the kernel $k$ on $\tilde{\mathcal{X}}$. Then the kernel $k(A(x),A(x'))$ is a kernel on $\mathcal{X}$.

**Lemma(Products of kernels are kernels)**. Given $k_1$ on $\mathcal{X}_1$ and $k_2$ on $\mathcal{X}_2$, then $k_1\times k_2$ is a kernel on $\mathcal{X}_1\times\mathcal{X}_2$. If $\mathcal{X}_1=\mathcal{X}_2=\mathcal{X}$, then $k=k_1\times k_2$ is a kernel on $\mathcal{X}$.



**Lemma(Polynomial kernels)**. Let $x,x'\in \mathbb{R}^d$ for $d\ge 1$， and let $m\ge 1$ be an integer and $c\ge 0$ be a positive real. then

$$
k(x,x')=(\langle x,x'\rangle+c)^m
$$

is a valid kernel.

&emsp;&emsp;Can we extends this combination of sum and product rule to sums with infinitely many terms? It turn s out we we can, as long as these don't blow up.

 **Definition** The space $\ell_p$  of $p$-summable sequences is defined as all sequences $(a_i)_{i\ge 1}$ for which

 $$
 \sum_{i=1}^{\infty}a_i^p <\infty
 $$

&emsp;&emsp;Kernels can be defined in terms of sequences in $\ell_2$.

 **Lemma**. Given a non-empty set $\mathcal{X}$, and a sequence of functions $(\phi_i(x))_{i\ge 1}$ in $\ell_2$ where $\phi_i :\mathcal{X}\rightarrow \mathbb{R}$ is the $i$th coordinate of the feature map $\phi(x)$. Then

 $$
 k(x,x')=\sum_{i=1}^\infty\phi_i(x)\phi_i(x')
 $$
 is a well-defined kernel on $\mathcal{X}$.

&emsp;&emsp;Taylor series expansions may be used to define kernels that have infinityly many features.

**Definition(Taylor series kernel)**. Assume we can define the Taylor series

$$
f(z)=\sum_{n=0}^\infty a_n z^n\quad |z|<r,z\in\mathbb{R}
$$

for $r\in (0,\infty]$, with $a_n\ge 0$ for all $n\ge 0$. Define $\mathcal{X}$ to be the $\sqrt{r}$-ball in $\mathbb{R}^d$, Then for $x,x'\in\mathbb{R}^d$ such that $\Vert x\Vert <\sqrt{r}$, we have the kernel

$$
k(x,x')=f(\langle x,x'\rangle)=\sum_{n=0}^\infty a_n\langle x,x'\rangle^n
$$

 **proof**. Non-negative weighted sums of kernels are kernels, and products of kernels are kernels, so the following is a kernel if it converges,

 $$
 k(x,x')=\sum_{n=1}^\infty a_n(\langle x,x'\rangle)^n
$$

 We have by Cauchy-Schwarz that

 $$
  |\langle x,x'\rangle |\le \Vert x\Vert\Vert x'\Vert<r
 $$

so the Taylor series converges.

&emsp;&emsp;An example of a Taylor series kernel is the exponential.

 **Example(Exponential kernel)**. The exponential kernel on $\mathbb{R}^d$ is defined as 

 $$
 k(x,x')=\exp(\langle x,x'\rangle)
 $$

&emsp;&emsp;We may conbine all the results above to obtain the following(product rule, mapping rule, et al.)

**Example(Gaussian kernel)**. The Gaussian kernel on $\mathbb{R}^d$ is defined as 

 $$
 k(x,x')=\exp(-\gamma^{-2}\Vert x-x'\Vert^2)
 $$

 proof. 

 $$
 \begin{split}
 k_1(x,x')&=\exp\left\{\frac{\langle x,x'\rangle}{\gamma^2}\right\}\\
 \underbrace{k_1(x,x')}_{\textrm{normalize}}&=\frac{\exp\{\frac{\langle x,x'\rangle}{\gamma^2}\}}{  \sqrt{\exp\left\{\frac{\Vert x\Vert^2}{\gamma^2} \right\}    \exp\left\{\frac{\Vert x'\Vert^2}{\gamma^2} \right\}}     }\\
 &=\exp\left\{{\frac{2\langle x,x'\rangle}{\gamma^2}-\frac{\langle x,x\rangle}{\gamma^2}-\frac{2\langle x',x'\rangle}{\gamma^2}}\right\}\\
 &=\exp(-\gamma^{-2}\Vert x-x'\Vert^2)\\\\
 &=k(x,x')
 \end{split}
 $$


## 核函数的基本运算

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
