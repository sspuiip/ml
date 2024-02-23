# 最大均值差

## 均值嵌入 (Mean Embedding)

&emsp;&emsp;若给定一集合$\mathcal{X}$的一个Borel概率测度$P$，则概率分布$P$的特征映射为，

$$
\mu_P=(...,\mathbb{E}_P[\varphi_i(x)],...)
$$

对于正定函数$k(x,y)$,$x\sim P, y\sim Q$,则有，

$$
\langle \mu_P,\mu_Q\rangle_\mathcal{F}=\mathbb{E}_{P,Q}[k(x,y)]
$$

&emsp;&emsp;**例**. 对于有限维内积空间$\mathcal{F}$，可以定义关于内积的期望，

$$
\varphi(x)=k(\cdot,x)=\begin{bmatrix} x\\ x^2\end{bmatrix},\quad f(\cdot)=\begin{bmatrix} a\\ b \end{bmatrix}
$$

则有，

$$
f(x)=\langle f(\cdot),\varphi(x)\rangle_{\mathcal{F}}=\begin{bmatrix} a \\ b\end{bmatrix}^\top\begin{bmatrix} x\\ x^2 \end{bmatrix}=ax+bx^2
$$

&emsp;&emsp;若有随机变量$x\sim P$，则

$$
\mathbb{E}_P[f(x)]=\mathbb{E}_P\left(\begin{bmatrix} a \\ b\end{bmatrix}^\top\begin{bmatrix} x\\ x^2 \end{bmatrix} \right)=\begin{bmatrix} a \\ b\end{bmatrix}^\top\begin{bmatrix} \mathbb{E}_P[x]\\ \mathbb{E}_P[x^2] \end{bmatrix}\triangleq \begin{bmatrix} a \\ b\end{bmatrix}^\top\begin{bmatrix} \mathbb{E}_P[\varphi_1(x)]\\ \mathbb{E}_P[\varphi_2(x)] \end{bmatrix} =\langle f,\mu_P\rangle_\mathcal{F}
$$

&emsp;&emsp;上述$\mu_P$即为**均值嵌入(Mean Embedding)**。若是无穷维，均值嵌入还存在吗？也就是说，是否存在一个$\mu_P\in\mathcal{F}$，使得$\mathbb{E}_P[f(x)]=\langle f,\mu_P\rangle_\mathcal{F}=\langle f,\mathbb{E}_P[\varphi(x)]\rangle_\mathcal{F}$成立？

>**定理 (均值嵌入存在性)**. 如果$\mathbb{E}_P \sqrt{k(x,x)}<\infty$，则$\exists \mu_P \in \mathcal{F} $.

&emsp;&emsp;**证明**. 线性算子$T_Pf\triangleq \mathbb{E}_P[f(x)]$，$\forall f\in \mathcal{F}$是有界的。因为，

$$
\begin{split}
|T_Pf|&=|\mathbb{E}_P[f(x)]|\\
&\le \mathbb{E}_P|f(x)|\\
&=\mathbb{E}_P|\langle f,\varphi(x)\rangle_\mathcal{F}|\\
&\le\mathbb{E}_P\left(\sqrt{k(x,x)}\cdot\Vert f\Vert_\mathcal{F} \right)
\end{split}
$$

因此，通过Riesz定理($\lambda_{T_P}=\mathbb{E}_P\sqrt{k(x,x)}$)可知，$\exists \mu_P\in\mathcal{F}$满足$T_P f=\langle f,\mu_P\rangle_\mathcal{F}$。

&emsp;&emsp;注意：Riesz定理是指在Hilbert空间$\mathcal{F}$，所有有界的线性算子$A$可以表示为$\langle \cdot,g_A\rangle_\mathcal{F}$对于某些$g_A\in \mathcal{F}$，即$Af=\langle f,g_A\rangle_\mathcal{F}$。同时，有界线性算子是指满足$Af\le \lambda_A \Vert f\Vert_\mathcal{F}$条件。

---

## 最大均值差分(Maximum Mean Discrepancy)

>**定义**. 最大均值差分(Maximum Mean Discrepancy)是指特征均值之间的距离。

$$
\begin{split}
\textrm{MMD}^2(P,Q)&=\Vert \mu_P -\mu_Q\Vert_\mathcal{F}^2\\
&=\underbrace{\mathbb{E}_P k(x,x')+\mathbb{E}_Q k(y,y')}_{\textrm{within distrib. similarity}}-\underbrace{2\mathbb{E}_{P,Q}k(x,y)}_{\textrm{cross-distrib. similarity}}
\end{split}
$$

&emsp;&emsp;**证明**.

$$
\begin{split}
\textrm{MMD}^2(P,Q)&=\Vert \mu_P -\mu_Q\Vert_\mathcal{F}^2\\
&=\langle \mu_P-\mu_Q,\mu_P-\mu_Q\rangle_\mathcal{F}\\
&=\langle\mu_P,\mu_P\rangle+\langle\mu_Q,\mu_Q\rangle-2\langle\mu_P,\mu_Q\rangle\\
&=\underbrace{\mathbb{E}_P[\mu_P(x)]}_{\mathbb{E}_P[f(x)]=\langle f,\varphi(x)\rangle_\mathcal{F}}+...\\
&=\underbrace{\mathbb{E}_P[\langle\mu_P,k(\cdot,x)\rangle]}_{\mu_P\triangleq(...,\mathbb{E}_P[\varphi_i(x)],...)}+...\\
&=\mathbb{E}_P[k(x,x')]+\mathbb{E}_Q[k(y,y')]-2\mathbb{E}_{P,Q}[k(x,y)]\\

\end{split}
$$

---

## 一种积分概率度量

### 积分概率度量(Integral Probability Metric, IPM)

&emsp;&emsp;IPM主要用于衡量两个概率分布之间的距离(相似性)。IPM寻求某种限制条件的函数集$\mathcal{F}$中的连续函数$f$，使得该函数能够提供更多的矩信息，然后寻找最优$f$使得概率分布$P(x)$与$Q(x)$之间的差异最大，该最大差异即为两分布之间的距离，即

$$
d_{\mathcal{F}}(P,Q)=\sup_{f\in\mathcal{F}}\mathbb{E}_P[f(x)]-\mathbb{E}_Q[f(x)]\triangleq\textrm{IPM}(P,Q)
$$

&emsp;&emsp;选择不同的函数空间$\mathcal{F}$，会导致IPM具有不同形式。如

:::{table} 常见积分概率度量
:width: 750px
:align: center
:widths: grid
| Name    | Formula    | Condition    |
| :---: | :--- | :--- |
| **Dudley**    |  $D_{bl}(P,Q)$$=\sup\limits_{\Vert f\Vert_{bl}\le q}\mathbb{E}_P[f]-\mathbb{E}_Q[f]$   |  $\mathcal{F}=\{ f: \Vert f\Vert_{bl}\le 1 \}$ , $\Vert f\Vert_{bl}=\Vert f\Vert_{\infty}+\Vert f\Vert_L$,  $\Vert f\Vert_{\infty}=\sup\{\vert f(x)\vert:x\in M\}$, $\Vert f\Vert_L=\sup\left\{\frac{\vert f(x)-f(y)\vert} {\rho(x,y)} : x\neq y\in M\right\}  $|
|**Wasserstein** |$W(P,Q)$$=\inf\limits_{\mu\in\mathcal{L}(P,Q)}\int\rho(x,y)d\mu(x,y) $|  $\mathcal{F}=\{ f:\Vert f\Vert_L \le 1 \}$ and $M$ is separable.       |
| **Maximum Mean Discrepancy** | $\textrm{MMD}(P,Q)$$=\Vert \mu_P-\mu_Q\Vert_{\mathcal{H}}^2$ | $\mathcal{F}=\{f:\Vert f\Vert_\mathcal{H}\le 1 \}$,$\mathcal{H}$为RKHS，$k$为其再生核。|

:::

### MMD积分概率度量

&emsp;&emsp;MMD没有直接定义函数空间$\mathcal{F}$，而是通过核方法在内积空间间接定义了连续函数$f$，使得$f$可以隐式计算所有阶次的统计量。MMD将$f$限制在再生核希尔伯特空间的单位球内$\Vert f\Vert_{\mathcal{H}}\le 1$。

> **定义**. MMD积分概率度量定义为，

$$
\textrm{MMD}(f,P,Q)\triangleq\sup_{\Vert f\Vert_\mathcal{H}\le 1}\mathbb{E}_P[f(x)]-\mathbb{E}_Q[f(x)]
$$

&emsp;&emsp;由Riesz定理可知，

$$
\begin{split}
\textrm{MMD}(f,P,Q)&=\sup_{\Vert f\Vert_\mathcal{H}\le 1}\langle f,\mathbb{E}_P[\varphi(x)]\rangle_\mathcal{H}-\langle f,\mathbb{E}_Q[\varphi(x)]\rangle_\mathcal{H}\\
&=\sup_{\Vert f\Vert_\mathcal{H}\le 1}\langle f,\mathbb{E}_P[\varphi(x)]-\mathbb{E}_Q[\varphi(x)]\rangle_\mathcal{H}
\end{split}
$$

两向量内积的最大值必定在两向量同方向上最得，即，

$$
f^*=\frac{\mathbb{E}_P[\varphi(x)]-\mathbb{E}_Q[\varphi(x)]}{\Vert \mathbb{E}_P[\varphi(x)]-\mathbb{E}_Q[\varphi(x)]\Vert_\mathcal{H}}
$$

代入原式即可得，

$$
\boxed{\textrm{MMD}(f,P,Q)=\Vert\mu_P-\mu_Q \Vert_\mathcal{H}}
$$

&emsp;&emsp;应用蒙特卡罗估计，可以计算出MMD。

$$
\begin{split}
\textrm{MMD}^2(f,P,Q)&=\Vert\mu_P-\mu_Q \Vert_\mathcal{H}^2\\
&\approx\left\Vert\frac1m\sum_{i=1}^m\varphi(x_i)-\frac1n\sum_{j=1}\varphi(x_j) \right\Vert_\mathcal{H}^2\\
&\approx \boxed{\frac{1}{m^2}\sum_{i=1}^m\sum_{i'=1}^m k(x_i,x_{i'})+\frac{1}{n^2}\sum_{j=1}^n\sum_{j'=1}^m k(x_j,x_{j'})-\frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^n k(x_i,x_j)}
\end{split}
$$
