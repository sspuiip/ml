# 矩阵性能指标

一个$m\times n$的矩阵可以看成是一种具有$mn$个元素的多变量。如果需要使用一个标量来概括多变量，可以使用矩阵的性能指标来表示。矩阵的性能指标一般有：二次型、行列式、特征值、迹和秩等。

## 二次型

任意方阵$\pmb{A}$的**二次型**为$\pmb{x}^\top \pmb{A}\pmb{x}$，其中$\pmb{x}$为任意非零向量。

$$
\begin{equation}
\begin{split}
\pmb{x}^\top \pmb{A}\pmb{x}&=\sum_i\sum_j a_{ij}x_ix_j\\
&=\sum_i a_{ii}x_i^2 +\sum_i^{n-1}\sum_{j=i+1}^n(a_{ij}+a_{ji})x_ix_j
\end{split}
\end{equation}
$$

如果将大于0的二次型$\pmb{x}^\top \pmb{A}\pmb{x}$称为**正定的二次型**，则矩阵$\pmb{A}$称为**正定矩阵**，即

$$
\forall \pmb{x}\neq 0,\quad \pmb{x}^\top \pmb{A}\pmb{x}>0
$$

成立。根据二次型的计算结果，可以进一步区分以下矩阵类型。

|  矩阵类型  |         标记          |                            二次型                            |
| :--------: | :-------------------: | :----------------------------------------------------------: |
|  正定矩阵  |  $\pmb{A}\succ 0$  | $\forall \pmb{x}\neq 0,\quad \pmb{x}^\top \pmb{A}\pmb{x}>0$ |
| 半正定矩阵 | $\pmb{A}\succeq 0$ | $\forall \pmb{x}\neq 0,\quad \pmb{x}^\top \pmb{A}\pmb{x}\ge 0$ |
|  负定矩阵  |  $\pmb{A}\prec 0$  | $\forall \pmb{x}\neq 0,\quad \pmb{x}^\top \pmb{A}\pmb{x}<0$ |
| 半负定矩阵 | $\pmb{A}\preceq 0$ | $\forall \pmb{x}\neq 0,\quad \pmb{x}^\top \pmb{A}\pmb{x}\le 0$ |
|  不定矩阵  |                       | $\forall \pmb{x}\neq 0,\quad \pmb{x}^\top \pmb{A}\pmb{x}$既有正值又有负值 |


### 常用性质

&emsp;&emsp;若记$\pmb{X}=(\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_n)$,其中$\pmb{x}_i \in \mathbb{R}^d$，则有，

$$
\sum_i \pmb{x}_i^\top\pmb{A}\pmb{x}_i =\textrm{tr}(\pmb{X}^\top\pmb{AX})
$$

特别地，若$\pmb{A}=\pmb{\Lambda}$为对角阵，则有，

$$
\sum_i \pmb{x}_i^\top\pmb{A}\pmb{x}_i =\textrm{tr}(\pmb{X}^\top\pmb{\Lambda X})=\sum_i \pmb{x}_i^\top \Lambda_{ii}\pmb{x}_i
$$


## 行列式

### 定义

一个$n\times n$的方阵的**行列式**记为$\det(\pmb{A})$或$|\pmb{A}|$，其定义为，


$$
\det(\pmb{A})=\left\lvert \begin{array}{cccc}a_{11}&a_{12}&\cdots & a_{1n}\\ a_{21}&a_{22}&\cdots & a_{2n}\\ \vdots&\vdots&\vdots & \vdots\\a_{n1}&a_{n2}&\cdots & a_{nn}\\ \end{array} \right\rvert
$$

行列式不为0的矩阵称为**非奇异矩阵**。

### 余子式

去掉矩阵第$i$行$j$列后得到的剩余行列式记为$A_{ij}$，称为矩阵元素$a_{ij}$的**余子式**。若去掉矩阵第$i$行$j$列后得到的剩余**子矩阵**记为$\pmb{A}_{ij}$，则有，


$$
A_{ij}=(-1)^{i+j}\det(\pmb{A}_{ij})
$$


任意一个方阵的行列式等于其任意行（列）元素与相对应余子式乘积之和，即，


$$
\begin{split}
\det(\pmb{A})&=a_{i1}A_{i1}+\cdots+a_{in}A_{in}=\sum_j a_{ij}\cdot(-1)^{i+j}\det(\pmb{A}_{ij})\\
&=a_{1j}A_{1j}+\cdots+a_{nj}A_{nj}=\sum_i a_{ij}\cdot(-1)^{i+j}\det(\pmb{A}_{ij})
\end{split}
$$

因此，行列式可以递推计算：$n$阶行列式由$n-1$阶行列式计算，由此递推直到$n=1$。



### 性质一

1. 行列式的两行（列）交换位置，则行列式的值不变，但符号改变。

2. 若矩阵的行（列）向量线性相关，则行列式为0。

3. $\det(\pmb{A})=\det(\pmb{A}^\top)$

4. $\det(\pmb{AB})=\det(\pmb{A})\det(\pmb{B})$

5. $\det(c\pmb{A})=c^n\det(\pmb{A})$

6. 若$\pmb{A}$非奇异，则$\det(\pmb{A}^{-1})=1/\det(\pmb{A})$

7. 三角矩阵、对角矩阵的行列式等于主对角线元素乘积。

   
   $$
   \det(\pmb{A})=\prod_i^n a_{ii}
   $$
   
8. 分块矩阵。

   
   $$
   \pmb{A}非奇异\Leftrightarrow\det\left[\begin{array}{cc}\pmb{A}&\pmb{B}\\\pmb{C}&\pmb{D} \end{array}\right]=\det(\pmb{A})\det(\pmb{D}-\pmb{C}\pmb{A}^{-1}\pmb{B})
   $$

### 性质二

1. Cauchy-Schwartz不等式，若$\pmb{A,B}$都是$m\times n$矩阵，则

   
   $$
   \det(\pmb{A}^\top\pmb{B})^2\le \det(\pmb{A}^\top\pmb{A})\det(\pmb{B}^\top\pmb{B})
   $$
   
2. Hadamard不等式，对于$m\times m$矩阵$\pmb{A}$有，

   
   $$
   \det(\pmb{A})\le \prod_{i=1}^m\left(\sum_{j=1}^m |a_{ij}|^2 \right)^{1/2}
   $$
   
3. Fischer不等式，若$\pmb{A}_{m\times m}, \pmb{B}_{m\times n},\pmb{C}_{n\times n}$，则有，

   
   $$
   \det\left[\begin{array}{cc}\pmb{A}&\pmb{B}\\\pmb{B}^\top&\pmb{C} \end{array}\right]\le\det(\pmb{A})\det(\pmb{C})
   $$
   
4.  Minkowski不等式，若$\pmb{A}_{m\times m},\pmb{B}_{m\times m}$半正定，则有，


$$
   \sqrt[m]{\det(\pmb{A}+\pmb{B})}\ge\sqrt[m]{\det(\pmb{A})}+\sqrt[m]{\det(\pmb{B})}
$$


5. 正定矩阵$\pmb{A}$的行列式大于0。

6. 半正定矩阵$\pmb{A}$的行列式大于等于0。



## 矩阵内积

矩阵内积是指：


$$
\langle\pmb{A},\pmb{B}\rangle=vec(\pmb{A})^\top vec(\pmb{B})=tr (\pmb{A}^\top\pmb{B})
$$



## 矩阵范数

### 向量范数

1. $L_0$范数： $\lVert \pmb{x}\rVert_0\triangleq$非零元素的个数。是一种虚拟的范数，在稀疏表示中有作用。

2. $L_1$范数： $\lVert \pmb{x}\rVert_1\triangleq\sum_i^n |x_i|=|x_1|+|x_2|+\dots+|x_n|$。

3. $L_2$范数： $\lVert \pmb{x}\rVert_2\triangleq\left(\sum_i^n x_i^2\right)=(x_1^2+x_2^2+\dots+x_n^2)^{1/2}$。

4. $L_\infty$范数： $\lVert \pmb{x}\rVert_\infty\triangleq\max\{|x_1|+|x_2|+\dots+|x_n|\}$。

5. $L_p$范数：$\lVert \pmb{x}\rVert_p=\left(\sum_i x_i^p\right)^{1/p}$。

### 矩阵范数

矩阵范数是矩阵的实值函数，且满足以下条件（与向量空间范数的定义类似），

1. 非负性： $\lVert \pmb{A}\rVert\ge 0$，$\lVert \pmb{A}\rVert= 0$当且仅当$\pmb{A}=0$。
2. 正比例：$\lVert c\pmb{A}\rVert=|c|\cdot\lVert\pmb{A}\rVert$。
3. 三角不等式：$\lVert \pmb{A}+\pmb{B}\rVert\le\lVert \pmb{A}\rVert+\lVert\pmb{B}\rVert$。
4. $\lVert\pmb{AB}\rVert\le\lVert\pmb{A}\rVert\cdot\lVert\pmb{B}\rVert$

常见矩阵范数主要有三类：诱导范数、元素形式范数和Schatten范数。

### 诱导范数

假设有矩阵$\pmb{A}\in \mathbb{R}^{m\times n}$，则有以下诱导范数定义。其实是一个向量范数的变形。

1. 矩阵$\pmb{A}$的诱导范数为，

   


$$
\begin{split}
\lVert \pmb{A}\rVert_{(m,n)} &\triangleq \max\{\lVert \pmb{Ax} \rVert :\pmb{x}\in R^n, \lVert \pmb{x}\rVert=1  \}\\
&=\max\left\{\frac{\lVert \pmb{Ax}_{(m)} \rVert}{\lVert \pmb{x}_{(n)} \rVert}:\pmb{x}\in R^n, \lVert \pmb{x}\rVert=1\right\}
\end{split}
$$



2. 矩阵$\pmb{A}$的诱导p范数为，

   
   $$
   \lVert \pmb{A}\rVert_p\triangleq\max_{\pmb{x}\neq 0}\frac{\lVert\pmb{Ax}\rVert_p}{\lVert\pmb{x}\rVert_p}
   $$
   

   当取如下值时，

   - $p=1$

     
     $$
     \lVert \pmb{A}\rVert_1\triangleq\max_{1\le j\le n}\sum_i^m|a_{ij}|
     $$
     

     计算过程如下：$\lVert \pmb{Ax}\rVert_1=\lVert \sum_j^nx_j\pmb{a}_j\rVert_1\le\sum_j^n|x_j|\cdot\lVert\pmb{a}_j\rVert_1\le\max_{1\le j\le n}\sum_i^m|a_{ij}|$ 。$\pmb{a}_j$为矩阵$\pmb{A}$的第$j$列。该范式计算结果等于矩阵$\pmb{A}$最大绝对值和的列。

   - $p=\infty$

     
     $$
     \lVert \pmb{A}\rVert_\infty\triangleq\max_{1\le i\le m}\sum_j^n|a_{ij}|
     $$
     

     计算过程如下：$\lVert \pmb{Ax}\rVert_\infty=\max_{1\le i\le m}\{\sum_{j=1}^n |a_{ij}x_j| \}  \le \max_{1\le i\le m}\sum_{j=1}^n|x_j|\cdot |a_{ij}|\le\max_{1\le i\le m}\sum_{j=1}^n|a_{ij}|$ 。该范式计算结果等于矩阵$\pmb{A}$最大绝对值和的行。

   - $p=2$

     
     $$
     \lVert \pmb{A}\rVert_2\triangleq\sqrt{\lambda_{\max}(\pmb{A}^\top\pmb{A})}=\sigma_{\max}(\pmb{A})
     $$
     

     计算结果为矩阵$\pmb{A}$的最大奇异值。

   

             

### 元素形式范数

元素形式范数就是将$m\times n$矩阵按列堆栈成$mn\times 1$维的向量，然后再使用向量形式的范数定义。

- $p$-矩阵范数

$$
\lVert \pmb{A}\rVert_p = \left(\sum_{i=1}\sum_{j=1}|a_{ij}|^p\right)^{1/p}
$$

  1. $p=1$时，$\lVert \pmb{A}\rVert_1=\sum_i\sum_j |a_{ij}|$。
  2. $p=\infty$时，$\lVert \pmb{A}\rVert_\infty=\max_{ij} |a_{ij}|$。
  3. $p=2$时，$\lVert \pmb{A}\rVert_2= \left(\sum_{i=1}\sum_{j=1}|a_{ij}|^2\right)^{1/2}$。该范数也称之为Frobenius范数。并且有如下性质，

  $$
   \lVert \pmb{A}\rVert_2=\sqrt{\textrm{tr}(\pmb{A}^\top\pmb{A})}=\langle \pmb{A},\pmb{A}\rangle^{1/2}
  $$




### Schatten范数

Schatten范数定义在矩阵的奇异值之上，可用于解决各类低秩问题：压缩感知、低秩矩阵与张量恢复等。

#### 核范数

核范数(nuclear norm)是Schatten范数的特例。典型应用场景：核范数最小化等价秩最小化。由于核范数最小化问题是一个**凸优化**问题，所以这种等价可直接降低求解各类**低秩问题**的难度。

**定义1** (核范数). 给定任意矩阵$\pmb{A}\in \mathbb{R}^{m\times n}$, 以及$r=\min(m,n)$，且矩阵$\pmb{A}$的奇异值为$\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r$，则矩阵$\pmb{A}$的核范数为，

$$
\lVert \pmb{X}\rVert_*=\sigma_1+\sigma_2+\cdots+\sigma_r
$$

通过SVD分解，即$\pmb{X}=\pmb{U\Sigma V}^\top$，则有

$$
\begin{split}
\lVert \pmb{X} \rVert_* &= \mathrm{tr}\left(\sqrt{\pmb{X}^\top\pmb{X}}\right)\\
&=\mathrm{tr}(\pmb{\Sigma})
\end{split}
$$



- **核范数的偏导数**

$$
\begin{split}
d\lVert \pmb{X} \rVert_*&=d\mathrm{tr}(\pmb{\Sigma})\\
&=\mathrm{tr}d(\pmb{\Sigma})\\
&=\mathrm{tr}(\pmb{U}^\top d\pmb{X}\pmb{V})\\
&=\mathrm{tr}(\pmb{V}\pmb{U}^\top d\pmb{X})\\
\end{split}
$$

由此可知，

$$
\frac{\partial \lVert \pmb{X} \rVert_*}{\partial \pmb{X}}=\pmb{UV}^\top
$$

注意：$d\pmb{X}=\pmb{U}(d\pmb{\Sigma})\pmb{V}^\top$可得到，$d\pmb{\pmb{\Sigma}}=\pmb{U}^\top (d\pmb{X})\pmb{V}$。


#### Schatten范数

相比于核范数，Schatten范数多出了一个参数$p$。在众多低秩问题中，核范数最小化扮演着非常重要的角色，Schatten 范数在形式上比核范数更为灵活，也同样能应用于诸多[低秩问题](https://zhuanlan.zhihu.com/p/104402273)。可参考NeurIPS文章《Factor Group-Sparse Regularization for Efficient Low-Rank Matrix Recovery》[[pdf]](https://proceedings.neurips.cc/paper/2019/file/0fc170ecbb8ff1afb2c6de48ea5343e7-Paper.pdf)[[code]](https://github.com/udellgroup/Codes-of-FGSR-for-effecient-low-rank-matrix-recovery)。

**定义2** (Schatten范数). 给定任意矩阵$\pmb{A}\in \mathbb{R}^{m\times n}$, 以及$r=\min(m,n), p>0$，且矩阵$\pmb{A}$的奇异值为$\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r$，则矩阵$\pmb{A}$的Schatten范数为，

$$
\lVert \pmb{X}\rVert_{Sp}=(\sigma_1^p+\sigma_2^p+\cdots+\sigma_r^p)^{1/p}
$$

## 迹

矩阵的迹是指$n\times n$矩阵$\pmb{A}$的所有对角元素之和，记为$\textrm{tr}(\pmb{A})$，即


$$
\textrm{tr}(\pmb{A})=a_{11}+a_{22}+\dots+a_{nn}=\sum_i^n a_{ii}
$$

### 性质一

- $\textrm{tr}(c\pmb{A}\pm d\pmb{B})=\textrm{tr}(\pmb{A})\pm \textrm{tr}(\pmb{B})$
- $\textrm{tr}(\pmb{A}^\top)=\textrm{tr}(\pmb{A})^\top$
- $\textrm{tr}(\pmb{ABC})=\textrm{tr}(\pmb{BCA})=\textrm{tr}(\pmb{CAB})$
- $\pmb{x}^\top\pmb{A}\pmb{x}=\textrm{tr}(\pmb{x}^\top\pmb{A}\pmb{x})$，特别地，$\pmb{x}^\top\pmb{y}=\textrm{tr}(\pmb{yx}^\top)$
- $\textrm{tr}(\pmb{A})=\lambda_1+\lambda_2+\cdots+\lambda_n$所有特征值之和。
- $tr\left[\begin{array}{cc}\pmb{A}&\pmb{B}\\\pmb{C}&\pmb{D}\end{array}\right]=\textrm{tr}(\pmb{A})+\textrm{tr}(\pmb{D})$
- $\textrm{tr}(\pmb{A}^k)=\sum_i \lambda_i^k$


### 性质二

- $\textrm{tr}(\pmb{A}^2)\le \textrm{tr}(\pmb{A}^\top\pmb{A})$
- $\textrm{tr}((\pmb{A}+\pmb{B})(\pmb{A}+\pmb{B})^\top)\le 2[\textrm{tr}(\pmb{A}\pmb{A}^\top)+\textrm{tr}(\pmb{B}\pmb{B}^\top)]$
- 若$\pmb{A,B}$都是对称矩阵，则$\textrm{tr}(\pmb{AB})\le \frac12 \textrm{tr}(\pmb{A}^2+\pmb{B}^2)$
