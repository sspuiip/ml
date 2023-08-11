# 最大均值差 (Maximum Mean Discrepancy)

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

## 最大均值差分

&emsp;&emsp;所谓的最大均值差分(Maximum Mean Discrepancy)是指特征均值之间的距离。

$$
\begin{split}
\textrm{MMD}^2(P,Q)&=\Vert \mu_P -\mu_Q\Vert_\mathcal{F}^2\\
&=\mathbb{E}_P k(x,x')+\mathbb{E}_Q k(y,y')-2\mathbb{E}_{P,Q}k(x,y)
\end{split}
$$

