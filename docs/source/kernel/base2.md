# 核函数基础2

## 投影

&emsp;&emsp;**定义 （投影）**. 投影$P$是一个映射，且满足以下条件：

$$
P(\pmb{x})=P(P(\pmb{x}))\quad\wedge\quad\langle P(\pmb{x}),\pmb{x}-P(\pmb{x})\rangle=0
$$(vector-proj)

&emsp;&emsp;**定义 （正交投影）**. 投影$P$的正交投影$P^\perp$为，

$$
P^\perp(\pmb{x}) \triangleq \pmb{x}-P(\pmb{x})
$$(orthogonal-proj)

&emsp;&emsp;$\phi(\pmb{x})$在向量$\pmb{w}$上的**投影**$P_{\pmb{w}}(\phi(\pmb{x}))$为，

$$
P_{\pmb{w}}(\phi(\pmb{x}))=\frac{\langle\phi(\pmb{x}),\pmb{w}\rangle}{\Vert\pmb{w}\Vert}\cdot\frac{\pmb{w}}{\Vert\pmb{w}\Vert}=\frac{\langle\phi(\pmb{x}),\pmb{w}\rangle}{\Vert\pmb{w}\Vert^2}\cdot\pmb{w}
$$

如果$\pmb{w}$已单位化，则有，

$$
P_{\pmb{w}}(\phi(\pmb{x}))=\langle\phi(\pmb{x}),\pmb{w}\rangle\cdot\pmb{w}=\pmb{w}\cdot\langle\phi(\pmb{x}),\pmb{w}\rangle=\pmb{w}\pmb{w}^\top\phi(\pmb{x})
$$(feature-proj)

因此，**正交投影**$ P_{\pmb{w}}^\bot \phi(\pmb{x}) $ 为，

$$
P_{\pmb{w}}^\bot\phi(\pmb{x})=(\pmb{I}-\pmb{ww}^\top)\phi(\pmb{x})
$$(feature-orthogonal-proj)


## 最大协方差的投影

&emsp;&emsp;假设有两个多维随机变量$(\pmb{x},\pmb{y})$，以及各自的投影单位向量$\pmb{w}_x,\pmb{w}_y$，则可以将变量投至各自投影方向从而得到两个1维随机变量$\pmb{w}_x^\top\pmb{x},\pmb{w}_y^\top\pmb{y}$。这两个变量的协方差可计算如下，

$$
\hat{\mathbb{E}}[\pmb{w}_x^\top\pmb{x}\pmb{w}_y^\top\pmb{y}]=\hat{\mathbb{E}}[\pmb{w}_x^\top\pmb{x}\pmb{y}^\top\pmb{w}_y]=\pmb{w}_x^\top\hat{\mathbb{E}}[\pmb{x}\pmb{y}^\top]\pmb{w}_y=\pmb{w}_x^\top \pmb{C}_{xy}\pmb{w}_y
$$(proj-covar)

其中，

$$
\pmb{C}_{xy}\triangleq \hat{\mathbb{E}}[\pmb{x}\pmb{y}^\top]=\frac1n\sum_{i=1}^n\pmb{x}_i\pmb{y}_i^\top
$$(covar-def)

&emsp;&emsp;投影单位向量$\pmb{w}_x,\pmb{w}_y$取什么值才能最大化投影后的协方差{eq}`proj-covar`呢？即有以下问题，

$$
\begin{split}
\max\limits_{\pmb{w}_x,\pmb{w}_y}\quad &\pmb{w}_x^\top \pmb{C}_{xy}\pmb{w}_y\triangleq C(\pmb{w}_x,\pmb{w}_y)\\
\textrm{s.t.}\quad &\lVert\pmb{w}_x\rVert_2=\lVert\pmb{w}_y\rVert_2=1
\end{split}
$$(maxi-covar-proj)

&emsp;&emsp;若对$\pmb{C}_{xy}$进行SVD分解，可得到$\pmb{U,\Sigma,V^\top}=svd(\pmb{C}_{xy})$，则式{eq}`maxi-covar-proj`的解为，

$$
\pmb{w}_x=\pmb{u}_1,\quad\pmb{w}_y=\pmb{v}_1.
$$(sol-of-max-covar)

注意：$\lVert\pmb{Uw}\rVert=\lVert\pmb{w}\rVert$，且$\pmb{U,V}$都是正交矩阵，因此$\pmb{w}_x$可以表示为$\pmb{u}_x$的形式$\pmb{Uu}_x$。

### 对偶形式

&emsp;&emsp;如果我们不想计算$\pmb{C}_{xy}$的SVD分解，例如在核函数定义的特征空间，则可以利用$\pmb{C}_{xy}\pmb{C}_{xy}^\top$得到$\pmb{U}$以及$\pmb{C}_{xy}^\top\pmb{C}_{xy}$得到$\pmb{V}$。假设有，

$$
\pmb{C}_{xy}^\top\pmb{C}_{xy}=\frac{1}{n^2}\pmb{Y}^\top\pmb{XX}^\top\pmb{Y}=\frac{1}{n^2}\pmb{Y}^\top\pmb{K}_x\pmb{Y}
$$(dual-represent)

与核PCA类似，投影方向$\pmb{u}_j$如下，

$$
\pmb{u}_j=\frac{1}{\sigma_j}\pmb{C}_{xy}\pmb{v}_j
$$(proj-of-direction)

因此，新样本$\phi(\pmb{x})$在$\pmb{u}_j$上的投影为，

$$
\pmb{u}_j^\top\phi(\pmb{x})=\frac{1}{n\sigma_j}\pmb{v}_j^\top\pmb{Y}^\top\pmb{X}\phi(\pmb{x})=\sum_{i=1}^n\left(\frac{1}{n\sigma_j}\pmb{Yv}_j\right)\kappa(\pmb{x}_i,\pmb{x})
$$(new-sample-proj)

## 广义特征值

&emsp;&emsp;广义特征值问题如下，

$$
\boxed{\pmb{Aw}=\lambda\pmb{Bw}}
$$(generalised-eigenvalue)

其中，$\pmb{A},\pmb{B}$均为对称矩阵，此外，$\pmb{B}$是正定的。标准特征值问题是上述问题的特殊形式，即$\pmb{B}=\pmb{I}$。对于**广义Rayleigh商**(generalized Rayleigh quotient)来说，

$$
\rho(\pmb{w})=\frac{\pmb{w}^\top\pmb{Aw}}{\pmb{w}^\top\pmb{Bw}}
$$(generalised-Rayleigh)

广义特征值也是最大化广义Rayleigh商的解，即

$$
\begin{split}
\max\limits_{\pmb{w}}\quad &\pmb{w}^\top\pmb{Aw}\\
\textrm{s.t.}\quad &\pmb{w}^\top\pmb{Bw}=1
\end{split}
$$(max-generalised-rayleigh)

&emsp;&emsp;式{eq}`generalised-eigenvalue`可以通过左乘矩阵$\pmb{B}^{-1}$转化为标准特征值问题，即

$$
\pmb{B}^{-1}\pmb{Aw}=\lambda\pmb{w}
$$(transform-generalised-eigenvalue)

然而，需要注意的是：虽然$\pmb{A},\pmb{B}$都是对称矩阵，但$\pmb{B}^{-1}\pmb{A}$不一定是对称矩阵。$\pmb{B}$是正定的对称矩阵，因此，可以分解为$\pmb{B}^{1/2}\pmb{B}^{1/2}=\pmb{B}$（可通过特征值分解得到平方根）。

&emsp;&emsp;对式{eq}`generalised-eigenvalue`左乘$\pmb{B}^{-1/2}$，并令$\boxed{\pmb{w}=\pmb{B}^{-1/2}\pmb{v}}$，则可得到下标准特征值等式，

$$
\boxed{\pmb{B}^{-1/2}\pmb{AB}^{-1/2}\pmb{v}=\lambda\pmb{v}}
$$(transform-generalized-eigenvalue-normalise)

可以验证$\pmb{B}^{-1/2}\pmb{AB}^{-1/2}=(\pmb{B}^{-1/2}\pmb{AB}^{-1/2})^\top$，即结果矩阵为对称矩阵。对该矩阵使用特征值分解，即可得到正交的特征向量$\pmb{v}_i$及其对应的特征值$\lambda_i$，因此，式{eq}`generalised-eigenvalue`的解为，

$$
\boxed{\pmb{w}_i=\pmb{B}^{-1/2}\pmb{v}_i}
$$(solution-of-generalized-eigenvaleu)

&emsp;&emsp;**广义特征值的性质**:

1. 如果特征值不相同，则在$\pmb{A},\pmb{B}$定义的度量(metrics)中，特征向量具有以下正交性：
  - $\pmb{w}_i^\top\pmb{B}\pmb{w}_j=\delta_{ij}$，满足此性质的向量$\pmb{w}_i,\pmb{w}_j$也称为关于$\pmb{B}$的共轭向量。
  - $\pmb{w}_i^\top\pmb{A}\pmb{w}_j=\delta_{ij}\lambda_i$

2. 矩阵$\pmb{A}$可分解为：$\pmb{A}=\sum_{i=1}^n\lambda_i\pmb{Bw}_i\pmb{Bw}_i^\top$。



