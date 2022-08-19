### 矩阵性能指标

一个$m\times n$的矩阵可以看成是一种具有$mn$个元素的多变量。如果需要使用一个标量来概括多变量，可以使用矩阵的性能指标来表示。矩阵的性能指标一般有：二次型、行列式、特征值、迹和秩等。

#### 二次型

任意方阵$\mathbf{A}$的**二次型**为$\mathbf{x}^\top \mathbf{A}\mathbf{x}$，其中$\mathbf{x}$为任意非零向量。

$$
\begin{equation}
\begin{split}
\mathbf{x}^\top \mathbf{A}\mathbf{x}&=\sum_i\sum_j a_{ij}x_ix_j\\
&=\sum_i a_{ii}x_i^2 +\sum_i^{n-1}\sum_{j=i+1}^n(a_{ij}+a_{ji})x_ix_j
\end{split}
\end{equation}
$$

如果将大于0的二次型$\mathbf{x}^\top \mathbf{A}\mathbf{x}$称为**正定的二次型**，则矩阵$\mathbf{A}$称为**正定矩阵**，即

$$
\forall \mathbf{x}\neq 0,\quad \mathbf{x}^\top \mathbf{A}\mathbf{x}>0
$$

成立。根据二次型的计算结果，可以进一步区分以下矩阵类型。

|  矩阵类型  |         标记          |                            二次型                            |
| :--------: | :-------------------: | :----------------------------------------------------------: |
|  正定矩阵  |  $\mathbf{A}\succ 0$  | $\forall \mathbf{x}\neq 0,\quad \mathbf{x}^\top \mathbf{A}\mathbf{x}>0$ |
| 半正定矩阵 | $\mathbf{A}\succeq 0$ | $\forall \mathbf{x}\neq 0,\quad \mathbf{x}^\top \mathbf{A}\mathbf{x}\ge 0$ |
|  负定矩阵  |  $\mathbf{A}\prec 0$  | $\forall \mathbf{x}\neq 0,\quad \mathbf{x}^\top \mathbf{A}\mathbf{x}<0$ |
| 半负定矩阵 | $\mathbf{A}\preceq 0$ | $\forall \mathbf{x}\neq 0,\quad \mathbf{x}^\top \mathbf{A}\mathbf{x}\le 0$ |
|  不定矩阵  |                       | $\forall \mathbf{x}\neq 0,\quad \mathbf{x}^\top \mathbf{A}\mathbf{x}$既有正值又有负值 |



#### 行列式

##### 定义

一个$n\times n$的方阵的**行列式**记为$\det(\mathbf{A})$或$|\mathbf{A}|$，其定义为，


$$
\det(\mathbf{A})=\left\lvert \begin{array}{cccc}a_{11}&a_{12}&\cdots & a_{1n}\\ a_{21}&a_{22}&\cdots & a_{2n}\\ \vdots&\vdots&\vdots & \vdots\\a_{n1}&a_{n2}&\cdots & a_{nn}\\ \end{array} \right\rvert
$$

行列式不为0的矩阵称为**非奇异矩阵**。

##### 余子式

去掉矩阵第$i$行$j$列后得到的剩余行列式记为$A_{ij}$，称为矩阵元素$a_{ij}$的**余子式**。若去掉矩阵第$i$行$j$列后得到的剩余**子矩阵**记为$\mathbf{A}_{ij}$，则有，


$$
A_{ij}=(-1)^{i+j}\det(\mathbf{A}_{ij})
$$


任意一个方阵的行列式等于其任意行（列）元素与相对应余子式乘积之和，即，


$$
\begin{split}
\det(\mathbf{A})&=a_{i1}A_{i1}+\cdots+a_{in}A_{in}=\sum_j a_{ij}\cdot(-1)^{i+j}\det(\mathbf{A}_{ij})\\
&=a_{1j}A_{1j}+\cdots+a_{nj}A_{nj}=\sum_i a_{ij}\cdot(-1)^{i+j}\det(\mathbf{A}_{ij})
\end{split}
$$

因此，行列式可以递推计算：$n$阶行列式由$n-1$阶行列式计算，由此递推直到$n=1$。



##### 性质一

1. 行列式的两行（列）交换位置，则行列式的值不变，但符号改变。

2. 若矩阵的行（列）向量线性相关，则行列式为0。

3. $\det(\mathbf{A})=\det(\mathbf{A}^\top)$

4. $\det(\mathbf{AB})=\det(\mathbf{A})\det(\mathbf{B})$

5. $\det(c\mathbf{A})=c^n\det(\mathbf{A})$

6. 若$\mathbf{A}$非奇异，则$\det(\mathbf{A}^{-1})=1/\det(\mathbf{A})$

7. 三角矩阵、对角矩阵的行列式等于主对角线元素乘积。

   
   $$
   \det(\mathbf{A})=\prod_i^n a_{ii}
   $$
   
8. 分块矩阵。

   
   $$
   \mathbf{A}非奇异\Leftrightarrow\det\left[\begin{array}{cc}\mathbf{A}&\mathbf{B}\\\mathbf{C}&\mathbf{D} \end{array}\right]=\det(\mathbf{A})\det(\mathbf{D}-\mathbf{C}\mathbf{A}^{-1}\mathbf{B})
   $$

##### 性质二

1. Cauchy-Schwartz不等式，若$\mathbf{A,B}$都是$m\times n$矩阵，则

   
   $$
   \det(\mathbf{A}^\top\mathbf{B})^2\le \det(\mathbf{A}^\top\mathbf{A})\det(\mathbf{B}^\top\mathbf{B})
   $$
   
2. Hadamard不等式，对于$m\times m$矩阵$\mathbf{A}$有，

   
   $$
   \det(\mathbf{A})\le \prod_{i=1}^m\left(\sum_{j=1}^m |a_{ij}|^2 \right)^{1/2}
   $$
   
3. Fischer不等式，若$\mathbf{A}_{m\times m}, \mathbf{B}_{m\times n},\mathbf{C}_{n\times n}$，则有，

   
   $$
   \det\left[\begin{array}{cc}\mathbf{A}&\mathbf{B}\\\mathbf{B}^\top&\mathbf{C} \end{array}\right]\le\det(\mathbf{A})\det(\mathbf{C})
   $$
   
4.  Minkowski不等式，若$\mathbf{A}_{m\times m},\mathbf{B}_{m\times m}$半正定，则有，


$$
   \sqrt[m]{\det(\mathbf{A}+\mathbf{B})}\ge\sqrt[m]{\det(\mathbf{A})}+\sqrt[m]{\det(\mathbf{B})}
$$


5. 正定矩阵$\mathbf{A}$的行列式大于0。

6. 半正定矩阵$\mathbf{A}$的行列式大于等于0。



#### 矩阵内积

矩阵内积是指：


$$
\langle\mathbf{A},\mathbf{B}\rangle=vec(\mathbf{A})^\top vec(\mathbf{B})=tr (\mathbf{A}^\top\mathbf{B})
$$



#### 矩阵范数

##### 向量范数

1. $L_0$范数： $\lVert \mathbf{x}\rVert_0\triangleq$非零元素的个数。是一种虚拟的范数，在稀疏表示中有作用。

2. $L_1$范数： $\lVert \mathbf{x}\rVert_1\triangleq\sum_i^n |x_i|=|x_1|+|x_2|+\dots+|x_n|$。

3. $L_2$范数： $\lVert \mathbf{x}\rVert_2\triangleq\left(\sum_i^n x_i^2\right)=(x_1^2+x_2^2+\dots+x_n^2)^{1/2}$。

4. $L_\infty$范数： $\lVert \mathbf{x}\rVert_\infty\triangleq\max\{|x_1|+|x_2|+\dots+|x_n|\}$。

5. $L_p$范数：$\lVert \mathbf{x}\rVert_p=\left(\sum_i x_i^p\right)^{1/p}$。

##### 矩阵范数

矩阵范数是矩阵的实值函数，且满足以下条件（与向量空间范数的定义类似），

1. 非负性： $\lVert \mathbf{A}\rVert\ge 0$，$\lVert \mathbf{A}\rVert= 0$当且仅当$\mathbf{A}=0$。
2. 正比例：$\lVert c\mathbf{A}\rVert=|c|\cdot\lVert\mathbf{A}\rVert$。
3. 三角不等式：$\lVert \mathbf{A}+\mathbf{B}\rVert\le\lVert \mathbf{A}\rVert+\lVert\mathbf{B}\rVert$。
4. $\lVert\mathbf{AB}\rVert\le\lVert\mathbf{A}\rVert\cdot\lVert\mathbf{B}\rVert$

常见矩阵范数主要有三类：诱导范数、元素形式范数和Schatten范数。

##### 诱导范数

假设有矩阵$\mathbf{A}\in \mathbb{R}^{m\times n}$，则有以下诱导范数定义。其实是一个向量范数的变形。

1. 矩阵$\mathbf{A}$的诱导范数为，

   


$$
\begin{split}
\lVert \mathbf{A}\rVert_{(m,n)} &\triangleq \max\{\lVert \mathbf{Ax} \rVert :\mathbf{x}\in R^n, \lVert \mathbf{x}\rVert=1  \}\\
&=\max\left\{\frac{\lVert \mathbf{Ax}_{(m)} \rVert}{\lVert \mathbf{x}_{(n)} \rVert}:\mathbf{x}\in R^n, \lVert \mathbf{x}\rVert=1\right\}
\end{split}
$$



2. 矩阵$\mathbf{A}$的诱导p范数为，

   
   $$
   \lVert \mathbf{A}\rVert_p\triangleq\max_{\mathbf{x}\neq 0}\frac{\lVert\mathbf{Ax}\rVert_p}{\lVert\mathbf{x}\rVert_p}
   $$
   

   当取如下值时，

   - $p=1$

     
     $$
     \lVert \mathbf{A}\rVert_1\triangleq\max_{1\le j\le n}\sum_i^m|a_{ij}|
     $$
     

     计算过程如下：$\lVert \mathbf{Ax}\rVert_1=\lVert \sum_j^nx_j\mathbf{a}_j\rVert_1\le\sum_j^n|x_j|\cdot\lVert\mathbf{a}_j\rVert_1\le\max_{1\le j\le n}\sum_i^m|a_{ij}|$ 。$\mathbf{a}_j$为矩阵$\mathbf{A}$的第$j$列。该范式计算结果等于矩阵$\mathbf{A}$最大绝对值和的列。

   - $p=\infty$

     
     $$
     \lVert \mathbf{A}\rVert_\infty\triangleq\max_{1\le i\le m}\sum_j^n|a_{ij}|
     $$
     

     计算过程如下：$\lVert \mathbf{Ax}\rVert_\infty=\max_{1\le i\le m}\{\sum_{j=1}^n |a_{ij}x_j| \}  \le \max_{1\le i\le m}\sum_{j=1}^n|x_j|\cdot |a_{ij}|\le\max_{1\le i\le m}\sum_{j=1}^n|a_{ij}|$ 。该范式计算结果等于矩阵$\mathbf{A}$最大绝对值和的行。

   - $p=2$

     
     $$
     \lVert \mathbf{A}\rVert_2\triangleq\sqrt{\lambda_{\max}(\mathbf{A}^\top\mathbf{A})}=\sigma_{\max}(\mathbf{A})
     $$
     

     计算结果为矩阵$\mathbf{A}$的最大奇异值。

   

             

##### 元素形式范数

元素形式范数就是将$m\times n$矩阵按列堆栈成$mn\times 1$维的向量，然后再使用向量形式的范数定义。

- $p$-矩阵范数

$$
\lVert \mathbf{A}\rVert_p = \left(\sum_{i=1}\sum_{j=1}|a_{ij}|^p\right)^{1/p}
$$

  1. $p=1$时，$\lVert \mathbf{A}\rVert_1=\sum_i\sum_j |a_{ij}|$。
  2. $p=\infty$时，$\lVert \mathbf{A}\rVert_\infty=\max_{ij} |a_{ij}|$。
  3. $p=2$时，$\lVert \mathbf{A}\rVert_2= \left(\sum_{i=1}\sum_{j=1}|a_{ij}|^2\right)^{1/2}$。该范数也称之为Frobenius范数。并且有如下性质，

  $$
   \lVert \mathbf{A}\rVert_2=\sqrt{tr(\mathbf{A}^\top\mathbf{A})}=\langle \mathbf{A},\mathbf{A}\rangle^{1/2}
  $$




##### Schatten范数

Schatten范数定义在矩阵的奇异值之上，可用于解决各类低秩问题：压缩感知、低秩矩阵与张量恢复等。

###### 核范数

核范数(nuclear norm)是Schatten范数的特例。典型应用场景：核范数最小化等价秩最小化。由于核范数最小化问题是一个凸优化问题，所以这种等价可直接降低求解各类低秩问题的难度。

**定义1** (核范数). 给定任意矩阵$\mathbf{A}\in \mathbb{R}^{m\times n}$, 以及$r=\min(m,n)$，且矩阵$\mathbf{A}$的奇异值为$\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r$，则矩阵$\mathbf{A}$的核范数为，

$$
\lVert \mathbf{X}\rVert_*=\sigma_1+\sigma_2+\cdots+\sigma_r

$$

###### Schatten范数

相比于核范数，Schatten范数多出了一个参数$p$。在众多低秩问题中，核范数最小化扮演着非常重要的角色，Schatten 范数在形式上比核范数更为灵活，也同样能应用于诸多[低秩问题](https://zhuanlan.zhihu.com/p/104402273)。可参考NeurIPS文章《Factor Group-Sparse Regularization for Efficient Low-Rank Matrix Recovery》.[[pdf]](https://proceedings.neurips.cc/paper/2019/file/0fc170ecbb8ff1afb2c6de48ea5343e7-Paper.pdf)[[code]](https://github.com/udellgroup/Codes-of-FGSR-for-effecient-low-rank-matrix-recovery)。

**定义2** (Schatten范数). 给定任意矩阵$\mathbf{A}\in \mathbb{R}^{m\times n}$, 以及$r=\min(m,n), p>0$，且矩阵$\mathbf{A}$的奇异值为$\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r$，则矩阵$\mathbf{A}$的Schatten范数为，

$$
\lVert \mathbf{X}\rVert_{Sp}=(\sigma_1^p+\sigma_2^p+\cdots+\sigma_r^p)^{1/p}
$$

#### 迹

矩阵的迹是指$n\times n$矩阵$\mathbf{A}$的所有对角元素之和，记为$tr(\mathbf{A})$，即


$$
tr(\mathbf{A})=a_{11}+a_{22}+\dots+a_{nn}=\sum_i^n a_{ii}
$$

##### 性质一

- $tr(c\mathbf{A}\pm d\mathbf{B})=tr(\mathbf{A})\pm tr(\mathbf{B})$
- $tr(\mathbf{A}^\top)=tr(\mathbf{A})^\top$
- $tr(\mathbf{ABC})=tr(\mathbf{BCA})=tr(\mathbf{CAB})$
- $\mathbf{x}^\top\mathbf{A}\mathbf{x}=tr(\mathbf{x}^\top\mathbf{A}\mathbf{x})$，特别地，$\mathbf{x}^\top\mathbf{y}=tr(\mathbf{yx}^\top)$
- $tr(\mathbf{A})=\lambda_1+\lambda_2+\cdots+\lambda_n$所有特征值之和。
- $tr\left[\begin{array}{cc}\mathbf{A}&\mathbf{B}\\\mathbf{C}&\mathbf{D}\end{array}\right]=tr(\mathbf{A})+tr(\mathbf{D})$
- $tr(\mathbf{A}^k)=\sum_i \lambda_i^k$


##### 性质二

- $tr(\mathbf{A}^2)\le tr(\mathbf{A}^\top\mathbf{A})$
- $tr((\mathbf{A}+\mathbf{B})(\mathbf{A}+\mathbf{B})^\top)\le 2[tr(\mathbf{A}\mathbf{A}^\top)+tr(\mathbf{B}\mathbf{B}^\top)]$
- 若$\mathbf{A,B}$都是对称矩阵，则$tr(\mathbf{AB})\le \frac12 tr(\mathbf{A}^2+\mathbf{B}^2)$
