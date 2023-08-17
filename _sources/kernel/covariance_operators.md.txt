# 协方差算子

&emsp;&emsp;在再生核希尔伯特空间理论中，协方差算子是一种很重要且广泛应用的工具。类似于协方差矩阵的无穷维版本。首先介绍Hilbert-Schmidt算子，然后引入协方差算子，并解释其亦是一种Hilbert-Schmidt算子。

## Hilbert-Schmidt算子

&emsp;&emsp;若有可分离(separable)希尔伯特特征空间$\mathcal{F},\mathcal{G}$，且$(e_i)_{i\in I}, (f_j)_{j\in J}$分别为$\mathcal{F},\mathcal{G}$的一组正交基，其中索引集$I,J$是有限集或是可数集。若有紧致(compact)线性算子$L:\mathcal{G}\rightarrow \mathcal{F}, M:\mathcal{G}\rightarrow \mathcal{F}$，则可定义Hilbert-Schmidt范式.

>**定义 (Hilbert-Schmidt算子)**. 若有以下Hilbert-Schmidt范式

$$
\Vert L\Vert_{\textrm{HS}}^2\triangleq \sum_{j\in J}\Vert Lf_j\Vert_\mathcal{F}^2=\sum_{i\in I}\sum_{j\in J}|\langle Lf_j,e_i\rangle_{\mathcal{F}}|^2
$$

是有限的($<\infty$)，则称$L$为Hilbert-Schmidty算子。

>**定义 (Hilbert-Schmidt算子空间与内积)**. Hibert-Schmidty算子从$\mathcal{G}$映射至$\mathcal{F}$（即$\delta:\mathcal{G}\rightarrow\mathcal{F}$），并形成一个Hilbert空间，记为$\textrm{HS}(\mathcal{G},\mathcal{F})$，其内积定义为

$$
\langle L,M\rangle_{\textrm{HS}}=\sum_{j\in J}\langle Lf_j,Mf_j\rangle_\mathcal{F}.
$$

该内积定义与正交基的选择无关。

&emsp;&emsp;显然，根据上述内积定义可知，Hilbert-Schmidty范式可以通过内积计算得到。

$$
\begin{split}
\langle L,L\rangle_{\textrm{HS}}&=\sum_{j\in J}\underbrace{\langle Lf_j,Lf_j\rangle_\mathcal{F}}_{\mathcal{F}空间内积}\\
&=\sum_{j\in J}\underbrace{\Vert Lf_j\Vert_\mathcal{F}^2}_{\textrm{parseval's identity}}\\
&=\sum_{i\in I}\sum_{j\in J}|\langle Lf_j,e_i\rangle_\mathcal{F}|^2\\
&=\Vert L\Vert_{\textrm{HS}}^2
\end{split}
$$

>**定义 (内积2)**. 内积的另一种等价形式，

$$
\langle L,M\rangle_{\textrm{HS}}=\sum_{i\in I}\sum_{j\in J}\langle Lf_j,e_i\rangle_\mathcal{F}\langle Mf_j,e_i\rangle_\mathcal{F}
$$

&emsp;&emsp;下面开始验证两种定义的等价性。任何$\mathcal{F}$的元素都可以展开为正交基的形式，即

$$
Lf_j=\sum_{i\in I}\alpha_i^{(j)}e_i,\quad Mf_j=\sum_{i'\in I}\beta_{i'}^{(j)}e_{i'}
$$

代入原定义可知，

$$
\begin{split}
\langle L,M\rangle_{\textrm{HS}}&\triangleq\sum_{j\in J}\langle Lf_j,Mf_j\rangle_\mathcal{F}\\
&=\sum_{j\in J}\left\langle \sum_{i\in I}\alpha_i^{(j)}e_i, \sum_{i'\in I}\beta_{i'}^{(j)}e_{i'}\right\rangle_\mathcal{F}\\
&=\sum_{j\in J}\sum_{i\in I}\alpha_i^{(j)}\beta_{i'}^{(j)}\\
&=\sum_{i\in I}\sum_{j\in J}\langle Lf_j,e_i\rangle_\mathcal{F}\langle Mf_j,e_i\rangle_\mathcal{F}
\end{split}
$$

---

### Rank-one算子

&emsp;&emsp;给定$b\in\mathcal{G},a\in\mathcal{F}$，可以定义张量积(tensor product)$a\otimes b$为rank-one算子$:\mathcal{G}\rightarrow \mathcal{F}$,

$$
(b\otimes a)f\rightarrow \langle f,a\rangle_\mathcal{F}b
$$

这是标准外积的推广，即如果$a,b,f$都是向量，则有$(ba^\top)f=(a^\top f)b$成立。

&emsp;&emsp;该算子是Hilbert-Schmidt算子吗？下面来计算算子的范数，

$$
\begin{split}
\Vert a\otimes b\Vert_{\textrm{HS}}^2&=\sum_{j\in J}\Vert (a\otimes b)f_j\Vert_\mathcal{F}^2\\
&=\sum_{j\in J}\Vert a\langle b, f_j\rangle_\mathcal{G}\Vert_\mathcal{F}^2\\
&=\Vert a\Vert_\mathcal{F}^2\sum_{j\in J}|\langle b,f_j\rangle_\mathcal{G}|^2\\
&=\Vert a\Vert_\mathcal{F}^2\Vert b\Vert_\mathcal{G}^2 < \infty
\end{split}
$$

因此，该算子是一个Hilbert-Schmidt算子。

&emsp;&emsp;给定另一个Hilbert-Schmidt算子$L\in \textrm{HS}(\mathcal{G},\mathcal{F})$，则有如下结果，

$$
\boxed{\langle L,a\otimes b\rangle_{\textrm{HS}}=\langle a,Lb\rangle_{\mathcal{F}}}
$$

一个特别的结果是，

$$
\boxed{\langle u\otimes v,a\otimes b\rangle_{\textrm{HS}}=\langle u,a\rangle_\mathcal{F}\langle b,v\rangle_\mathcal{G}}
$$

&emsp;&emsp;**证明**. $b$可以由$\mathcal{G}$的正交基展开为，

$$
b=\sum_{j\in J}\langle b,f_j\rangle_\mathcal{G}f_j
$$

则有，

$$
\begin{split}
\langle a,Lb\rangle&=\left\langle a,L\left(  \sum_j \langle b, f_j\rangle_\mathcal{G} f_j\right)\right\rangle_\mathcal{F}\\
&=\left\langle a,\sum_j L\left(   \langle b, f_j\rangle_\mathcal{G} f_j\right)\right\rangle_\mathcal{F},\qquad L(f+g)=L(f)+L(g)\\
&=\left\langle a,\sum_j \langle b, f_j\rangle_\mathcal{G} L\left(    f_j\right)\right\rangle_\mathcal{F}, \qquad L(\alpha f)=\alpha L(f) \\
&=\sum_j \left\langle a,\langle b, f_j\rangle_\mathcal{G} L\left(    f_j\right) \right\rangle_\mathcal{F},\qquad \langle a,b+c\rangle=\langle a,b\rangle+\langle a,c\rangle \\
&=\sum_j\langle b,f_j\rangle_\mathcal{G}\langle a,Lf_j\rangle_\mathcal{F},\qquad \langle a,kb\rangle=k\langle a,b\rangle
\end{split}
$$

以及，

$$
\begin{split}
\langle a\otimes b,L\rangle_{\textrm{HS}}&=\sum_j\langle Lf_j,(a\otimes b)f_j\rangle_\mathcal{F}\\
&=\sum_j\langle Lf_j,\langle b,f_j\rangle_\mathcal{G}a\rangle_\mathcal{F},\qquad (a\otimes b)f_j=\langle b,f_j\rangle_\mathcal{G}a\\
&=\sum_j\langle b,f_j\rangle_\mathcal{G}\langle Lf_j,a\rangle_\mathcal{F}
\end{split}
$$

显然，上述两个等式相等是成立的。特别地，如果我们用$u\otimes v$代替$L$，则有，

$$
\langle a\otimes b,u\otimes v\rangle_{\textrm{HS}}=\langle a,(u\otimes v)b\rangle_\mathcal{F}
=\langle b,v\rangle_{\mathcal{G}}\langle a,u\rangle_\mathcal{F}
$$

---

### Cross-covariance算子

&emsp;&emsp;定义算子前，我们先假设$\mathcal{F},\mathcal{G}$为再生核Hilbert空间，其核分别为$k$和$l$，对应的特征映射分别为$\phi,\psi$。该算子主要作用是互协方差矩阵泛化到无穷维。所期望达到的特征映射类似于，

$$
\tilde{C}_{XY}=\mathbb{E}[xy^\top],\qquad f^\top\tilde{C}_{XYg}=\mathbb{E}_{XY}[(f^\top x)(g^\top y

)]
$$

其中$\tilde{C}_{XY}$未中心化，中心化的协方差算子为，

$$
C_{XY}=\tilde{C}_{XY}-\mu_X\mu_Y^\top
$$
