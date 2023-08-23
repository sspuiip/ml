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

&emsp;&emsp;给定$b\in\mathcal{G},a\in\mathcal{F}$，可以定义张量积$a\otimes b$(tensor product)为rank-one算子$a\otimes b:\mathcal{G}\rightarrow \mathcal{F}$,

$$
(a\otimes b)g\rightarrow \langle g,b\rangle_\mathcal{G}a
$$

这是标准外积的推广，即如果$a,b,g$都是向量，则有$(ab^\top)g=(b^\top g)a$成立。

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

#### Cross-covariance算子定义

&emsp;&emsp;定义算子前，我们先假设$\mathcal{F},\mathcal{G}$为再生核Hilbert空间，其核分别为$k$和$l$，对应的特征映射分别为$\phi,\psi$。该算子主要作用是互协方差矩阵泛化到无穷维。所期望达到的特征映射类似于，

$$
\tilde{C}_{XY}=\mathbb{E}[xy^\top],\qquad f^\top\tilde{C}_{XYg}=\mathbb{E}_{XY}[(f^\top x)(g^\top y)]
$$

其中$\tilde{C}_{XY}$未中心化，中心化的协方差算子为，

$$
C_{XY}=\tilde{C}_{XY}-\mu_X\mu_Y^\top
$$

&emsp;&emsp;注意：$\phi(x)\otimes \psi(y)$是$\textrm{HS}(\mathcal{G},\mathcal{L})$空间的一个随机变量（已证明$\forall A\in\textrm{HS}(\mathcal{G},\mathcal{L})$,线性形式(linear form)$\langle\phi(x)\otimes\psi(y),A\rangle_{\textrm{HS}}$是可度量的）。为了使得该随机变量的期望存在，必要条件$\phi(x)\otimes\psi(y)$的范式(norm)有界，也就是$\mathbb{E}_{XY}[\Vert \phi(x)\otimes\psi(y)\Vert_{\textrm{HS}}]<\infty$。若给定期望(记为, $\tilde{C}_{XY}$)存在，则这个期望是某个成员且惟一，且满足，

$$
\boxed{\langle\tilde{C}_{XY},A\rangle_{\textrm{HS}}=\mathbb{E}_{XY}\langle\phi(x)\otimes\psi(y),A\rangle_{\textrm{HS}}}
$$

&emsp;&emsp;**证明**. 算子$T_{XY}$验证有界性。

$$
\begin{split}
T_{XY}: \textrm{HS}(\mathcal{G},\mathcal{F})&\rightarrow \mathbb{R}\\
A&\mapsto \mathbb{E}_{XY}\langle\phi(x)\otimes\psi(y),A\rangle_{\textrm{HS}}
\end{split}
$$

**当$\mathbb{E}_{XY}[\Vert\phi(x)\otimes \psi(y)\Vert_{\textrm{HS}}]<\infty$，可知算子$T_{XY}$是有界的。**

&emsp;&emsp;因为，

$$
\begin{split}
|\mathbb{E}_{XY}\langle\phi(x)\otimes\psi(y),A\rangle_{\textrm{HS}}|&\le\mathbb{E}_{XY}|\langle\phi(x)\otimes\psi(y),A\rangle_{\textrm{HS}}|\\
&\le\Vert A\Vert_{\textrm{HS}}\mathbb{E}_{XY}[\Vert \phi(x)\otimes \psi(y)\Vert_{\textrm{HS}}]
\end{split}
$$

通过Riesz表示定理可知，对于有界线性算子$T_{XY}$，总存在一个$\mathbb{E}_{XY}\langle \cdot,\phi(x)\otimes\psi(y)\rangle_{\textrm{HS}}$，使得$T_{XY}A$可以写成内积的形式$T_{XY}A=\mathbb{E}_{XY}\langle\phi(x)\otimes\psi(y),A\rangle_{\textrm{HS}}$。

&emsp;&emsp;上述条件可以继续简化，

$$
\begin{split}
\mathbb{E}_{XY}[\Vert \phi(x)\otimes \psi(y)\Vert_{\textrm{HS}}]&=\mathbb{E}_{XY}(\Vert \phi(x)\Vert_\mathcal{F}\Vert \psi(y)\Vert_\mathcal{G})\\
&=\mathbb{E}_{XY}\left(\sqrt{k(x,x)l(y,y)} \right)<\infty
\end{split}
$$

简化后的结果可以当作一个弱条件(weaker condition)来使用。这个条件是上述Jensen's不等式所隐含的。

#### 特例——$f\otimes g$

&emsp;&emsp;当$A=f\otimes g$时，

$$
\begin{split}
\langle f\otimes g,\tilde{C}_{XY}\rangle_{\textrm{HS}}&=\langle f,\tilde{C}_{XY}g\rangle_\mathcal{F}\\
&=\mathbb{E}_{XY}\langle \phi(x)\otimes \psi(y),f\otimes g\rangle_{\textrm{HS}}\\
&=\mathbb{E}_{XY}[\langle f,\phi(x)\rangle_\mathcal{F}\langle g,\psi(y)\rangle_\mathcal{G}]\\
&=\mathbb{E}_{XY}[f(x)g(y)]\\
&=\textrm{cov}(f(x),g(y)),\qquad \{\phi(x)=\phi(x)-\mathbb{E}_x\phi(x), \psi(y)=\psi(y)-\mathbb{E}_y\psi(y)\}
\end{split}
$$

---
## 约束协方差

&emsp;&emsp;一般来说，传统的协方差是指，若有随机变量$x,y$，则使用协方差可以判断两随机变量是否独立，即

$$
\textrm{cov}(x,y)=\mathbb{E}_{xy}[xy]-\mathbb{E}[x]\mathbb{E}[y]
$$

更一般地，已有工作证明，

>**定理**[1](https://proceedings.mlr.press/r5/gretton05a.html). 随机变量$x$与$y$是独立的当且仅当对于每一对连续有界算子$f,g$，都有$\textrm{cov}(f(x),g(y))=0$成立。

由此可以引出约束协方差(constrained covariance, COCO)的定义，

$$
\begin{split}
\textrm{COCO}(P_{XY})&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\textrm{cov}[f(x),g(y)]\\
&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\textrm{cov}\left[\left(\sum_{j=1}^\infty f_j\varphi_j(x)\right)\left(\sum_{j=1}^\infty g_j\phi_j(y)\right)  \right]
\end{split}
$$

若对特征映射中心化，即$\tilde{\varphi}(x)=\varphi(x)-\mathbb{E}_x\varphi(x), \tilde{\phi}(y)=\phi(y)-\mathbb{E}_y\phi(y)$，则COCO转变为以下形式，

$$
\begin{split}
\textrm{COCO}(P_{XY})&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\mathbb{E}_{xy}\left[\left(\sum_{j=1}^\infty f_j\tilde{\varphi}_j(x)\right)\left(\sum_{j=1}^\infty g_j\tilde{\phi}_j(y)\right)  \right]\\
&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\mathbb{E}_{xy}[\langle f,\tilde{\varphi}(x)\rangle_\mathcal{F},\langle g,\tilde{\phi}(y)\rangle_\mathcal{G}]\\
&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\mathbb{E}_{xy}\langle \tilde{\varphi}(x)\otimes\tilde{\phi}(y),f\otimes g \rangle_{\textrm{HS}} \\
&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\langle f,C_{\tilde{\varphi}(x)\tilde{\phi}(y)}g\rangle_\mathcal{F}\\
&=\sup\limits_{\Vert f\Vert_\mathcal{F}\le 1, \Vert g\Vert_\mathcal{G}\le 1}\begin{bmatrix}f_1\\f_2\\ \vdots\end{bmatrix}^\top\underbrace{\mathbb{E}_{xy}\left(\begin{bmatrix}\tilde{\varphi}_1(x) \\ \tilde{\varphi}_2(x)\\\vdots \end{bmatrix} \begin{bmatrix}\tilde{\phi}_1(y) & \tilde{\phi}_2(y)&\cdots \end{bmatrix}\right)}_{C_{\tilde{\varphi}(x)\tilde{\phi}(y)}}\begin{bmatrix}g_1\\g_2\\ \vdots\end{bmatrix}
\end{split}
$$

可以看出，COCO即为$C_{\tilde{\varphi}(x)\tilde{\phi}(y)}$的最大奇异值。

### COCO的估计

&emsp;&emsp;给定样本集$\{(x_i,y_i)\}_{i=1}^n\sim P_{XY}$，COCO的经验估计${\textrm{COCO}_{\textrm{emp}}}$(empirical)怎么计算？${\textrm{COCO}_{\textrm{emp}}}$是下式的最大特征值$\gamma_{\textrm{max}}$，

$$
\begin{bmatrix}0 & \frac1n\tilde{K}\tilde{L}\\\frac1n\tilde{L}\tilde{K}&0 \end{bmatrix}\begin{bmatrix}\alpha\\\beta \end{bmatrix}=\gamma\begin{bmatrix}\tilde{K}&0\\0&\tilde{L}\end{bmatrix}\begin{bmatrix}\alpha\\\beta \end{bmatrix}
$$

&emsp;&emsp;其中，$\tilde{K}_{ij}=\langle \phi(x_i)-\hat{\mu}_x,\phi(x_j)-\hat{\mu}_x\rangle_\mathcal{F}\triangleq\langle\tilde{\phi}(x_i),\tilde{\phi}(x_j)\rangle_\mathcal{F}$，以及$\tilde{L}_{ij}=\langle\tilde{\psi}(y_i),\tilde{\psi}(y_j)\rangle_\mathcal{G}$。

**proof**. 

1. 首先，协方差算子可以根据下式估计，

$$
\hat{C}_{XY}\triangleq\frac1n\sum_{i=1}^n\phi(x_i)\otimes\psi(y_i)-\hat{\mu}_x\otimes\hat{\mu}_y;\quad\hat{\mu}_x=\frac1n\sum_i^n\phi(x_i)
$$

引入中心化矩阵$H=I_n-n^{-1}11^\top$，则有

$$
\hat{C}_{XY}=\frac1n XHY^\top
$$

其中，

$$
X=\begin{bmatrix}\phi(x_1)&\phi(x_2)&\cdots&\phi(x_n)\end{bmatrix};\quad Y=\begin{bmatrix}\psi(y_1)&\psi(y_2)&\cdots&\psi(y_n)\end{bmatrix}
$$

1. 构造拉格朗日函数，

$$
\mathcal{L}(f,g,\lambda,\gamma)=-f^\top \hat{C}_{XY} g+\frac{\lambda}{2}\left(\Vert f\Vert_\mathcal{F}^2-1 \right)+\frac{\gamma}{2}\left(\Vert g\Vert_\mathcal{G}^2-1\right)
$$

其中，

$$
f=\sum_{i=1}^n\alpha_i[\phi(x_i)-\hat{\mu}_x]=XH\alpha; \quad g=YH\beta
$$

容易得到，

$$
f^\top \hat{C}_{XY} g=\frac1n\alpha^\top\tilde{K}\tilde{L}\beta;\quad \Vert f\Vert^2=\alpha^\top HXX^\top H\alpha=\alpha^\top\tilde{K}\alpha;\quad \Vert g\Vert^2_\mathcal{G}=\beta^\top \tilde{L}\beta
$$

整理后代入拉格朗日函数，

$$
\mathcal{L}(\alpha,\beta,\lambda,\gamma)=-\frac1n\alpha^\top\tilde{K}\tilde{L}\beta+\frac{\lambda}{2}(\alpha^\top\tilde{K}\alpha-1)+\frac{\gamma}{2}(\beta^\top \tilde{L}\beta-1)
$$

&emsp;&emsp;求偏导后整理得到，

$$
\begin{bmatrix}0 & \frac1n\tilde{K}\tilde{L}\\\frac1n\tilde{L}\tilde{K}&0 \end{bmatrix}\begin{bmatrix}\alpha\\\beta \end{bmatrix}=\gamma\begin{bmatrix}\tilde{K}&0\\0&\tilde{L}\end{bmatrix}\begin{bmatrix}\alpha\\\beta \end{bmatrix}
$$

即，$\gamma$最优值为上式最大特征值$\gamma_{\textrm{max}}$。