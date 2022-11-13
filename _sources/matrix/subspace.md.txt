### 子空间分析

#### 什么是子空间

##### 概念

&emsp;&emsp;**定义**  假设$S=\{\pmb{u}_1,...,\pmb{u}_m\}$是向量空间$V$的向量子集，则$\pmb{u}_1,...,\pmb{u}_m$的所有线性组合的集合$W$称为由$\pmb{u}_1,...,\pmb{u}_m$张成的**子空间**，即

$$
W=\mathrm{span}(\pmb{u}_1,...,\pmb{u}_m)=\{\pmb{u}|\pmb{u}=a_1\pmb{u}_1+...+a_m\pmb{u}_m\}
$$

其中$W$称为子空间的**张成集**，每个向量称为$W$的**生成元**。

##### 正交补

&emsp;&emsp;子空间$S_1,...,S_n$的交

$$
S=S_1\cap S_2\cap ... \cap S_n
$$

是所有子空间共有的向量组成的集合。

&emsp;&emsp;**定义** 若$S=\{\pmb{0}\}$，则称子空间$S_1,...,S_n$为**无交连**。

&emsp;&emsp;**定义** 无交连的子空间并集$S=S_1\cup S_2\cup ... \cup S_n$称为**子空间的直和**，记为，

$$
S=S_1 \oplus S_2 \oplus ... \oplus S_n
$$

&emsp;&emsp;**定义** 子空间$S$正交的所有向量的集合组成一个向量子空间，称为$S$的**正交补**空间，记为$S^\bot$。

&emsp;&emsp;1.  正交连的两个子空间不一定正交，但正交的两个子空间必定是无交连。

&emsp;&emsp;2.  正交补空间是比正交空间更严格的概念。

&emsp;&emsp;**定义** 若下式成立，则称子空间为相对于$\pmb{A}$不变的，

$$
\forall\pmb{x}\in S \quad\Rightarrow\quad \pmb{Ax}\in S
$$

##### 正交投影

&emsp;&emsp;假设有两个子空间$S,H$，如果有线性变换矩阵$\pmb{P}$，将$\pmb{x}\in\mathbb{R}^n$映射为子空间$S$的向量$\pmb{x}_1$，则这种线性变换称为沿着$H$的方向到$S$的**投影算子**，记为$\pmb{P}_{S|H}$。特别地，若$S,H$是正交补，则$\pmb{P}_{S|S^\bot}$是将$\pmb{x}$沿着与子空间$S$垂直方向的投影，称为子空间的**正交投影**，记为$\pmb{P}_S$。


&emsp;&emsp;假设两个子空间$S,H$其交集为空$\{\pmb{0}\}$，其直和空间为$V=S\oplus H$，则直和空间向量可分解为，

$$
\pmb{x}=\pmb{x}_s+\pmb{x}_h
$$

其中，$\pmb{x}_s\in S, \pmb{x}_h\in H$。由于$S\cap H=\{\pmb{0}\}$，则该分解唯一。定义一个沿着子空间$H$到子空间$S$的投影算子$\pmb{P}_{S|H}$，即，

$$
\pmb{Px}=\pmb{x}_s
$$

代入直和分解向量，则有，

$$
\pmb{x}_h=(\pmb{I}-\pmb{P})\pmb{x}
$$

即，$\pmb{I}-\pmb{P}$是沿着子空间$S$到子空间$H$的投影算子。因此，可知，

$$
\pmb{x}=\pmb{Px}+(\pmb{I}-\pmb{P})\pmb{x}
$$

以及，子空间的内部向量投影就是其本身，即$\pmb{Px}_1=\pmb{x}_1$,从而有，

$$
\pmb{P}^2\pmb{x}_1=\pmb{Px}_1=\pmb{x}_1
$$

也就是幂等性，

$$
\pmb{P}^2=\pmb{P}
$$

&emsp;&emsp;**定义** 若$Range(\pmb{P})=S,\pmb{P}^2=\pmb{P}$和$\pmb{P}^\top=\pmb{P}$，则矩阵$\pmb{P}$称为到子空间$S$的**正交投影**。

&emsp;&emsp;观察可知，线性变换矩阵，

$$
\pmb{P}_S=\pmb{A}(\pmb{A}^\mathrm{H}\pmb{A})^{-1}\pmb{A}^\mathrm{H}
$$

满足正交投影算子定义的幂等性和Hermitian性，且$\pmb{P}_S\pmb{A}=\pmb{A}$即满足$Range(\pmb{P})=S$ $=Span(\pmb{A})$。因此该矩阵是到由$\pmb{A}$的列向量生成的子空间$S$上的正交投影算子。

#### 列（行）空间与零空间

&emsp;&emsp;**定义 [列（行）空间]** 若$A=[\pmb{a}_1,...,\pmb{a}_n]\in\mathbb{C}^{m\times n}$为复矩阵，则列（行）向量的所有线性组合构成一个子空间，称为矩阵的列（行）空间。

$$
\begin{split}
\mathrm{Col}(A)&=\mathrm{Span}(\pmb{a}_1,...,\pmb{a}_n)\\
\mathrm{Row}(A)&=\mathrm{Span}(\pmb{r}_1,...,\pmb{r}_n)\\
\end{split}
$$

&emsp;&emsp;一般将$\mathrm{Span}(A)$作为$A$的列空间缩写。类似地，$\mathrm{Span}(A^H)$表示$A$的复共轭轩置矩阵$A^H$的列空间。由于$A^H$的列空间就是矩阵$A$的复共轭行向量，故有，

$$
\mathrm{Row}(A)=\mathrm{Col}(A^H)=\mathrm{Span}(A^H)=\mathrm{Span}(\pmb{r}_1,...,\pmb{r}_m)
$$

&emsp;&emsp;列（行）空间是直接针对矩阵$A_{m\times n}$本身定义的向量子空间。此外还有通过矩阵变换$\pmb{Ax}$定义的子空间：变换的值域和零空间。

&emsp;&emsp;**定义. [值域，零空间]** 若$\pmb{A}$是一个复矩阵，则$\pmb{A}$的值域(range)定义为，

$$
\mathrm{Range}(\pmb{A})=\{\pmb{y}\in\mathbb{C}^m | \pmb{Ax}=\pmb{y},\quad \pmb{x}\in\mathbb{C}^n\}
$$

矩阵$\pmb{A}$的零空间(null space)也称为$\pmb{A}$的核(kernel)，为齐次线性方程$\pmb{Ax=0}$的解向量的集合，

$$
\mathrm{Null}(A)=\mathrm{Ker}(A)=\{\pmb{x}\in\mathbb{C}^n|\pmb{Ax=0}\}
$$

&emsp;&emsp;显然，矩阵$\pmb{A}$的值域就是$\pmb{A}$的列空间，即，

$$
\mathrm{Range}(\pmb{A})=\{\pmb{y}\in\mathbb{C}^m | \pmb{Ax}=\pmb{y}=\sum_j x_j \pmb{a}_j,\quad \pmb{x}\in\mathbb{C}^n\}=\mathrm{Span}(\pmb{a}_1,...,\pmb{a}_n)
$$

&emsp;&emsp;**性质.**

&emsp;&emsp;1. $\mathrm{Range}(\pmb{A})=\mathrm{Col}(\pmb{A})=\mathrm{Span}(\pmb{a}_1,...,\pmb{a}_n)$

&emsp;&emsp;2. $\mathrm{Row}(\pmb{A})=\mathrm{Col}(\pmb{A}^H)=\mathrm{Range}(\pmb{A}^H)$

&emsp;&emsp;3. 零空间与行空间正交。$\mathrm{Row}(\pmb{A})^\bot=\mathrm{Null}(\pmb{A})$

&emsp;&emsp;4. $\mathrm{Col}(\pmb{A})^\bot=\mathrm{Null}(\pmb{A})$

##### 子空间基的构造

&emsp;&emsp;**定理.** 矩阵$\pmb{A}_{m\times n}$的列空间与行空间的维数相等。该维数就是$\pmb{A}$的秩$\mathrm{rank}(\pmb{A})$，且有如下关系，

$$
\mathrm{rank}(\pmb{A})+\mathrm{dim}[\mathrm{Null}(\pmb{A})]=n
$$

&emsp;&emsp;**定理.** 若$\pmb{A}_{m\times n}=\pmb{QR}$是一个列满秩矩阵的分解，则有，

$$
\mathrm{Span}(\pmb{a}_1,...,\pmb{a}_k)=\mathrm{Span}(\pmb{q}_1,...,\pmb{q}_k)\quad k=1,...,n
$$

&emsp;&emsp;下面介绍奇异值分解的基空间标准正交基的构造方法。任意一个矩阵$\pmb{A}_{m\times n}$可以分解为以下形式，

$$
\pmb{A}=\pmb{U\Sigma V^H}
$$

其中，$\pmb{U}=[\pmb{U}_r,\tilde{\pmb{U}}_r]$，$\pmb{V}=[\pmb{V}_r,\tilde{\pmb{V}}_r]$，以及，

$$
\pmb{\Sigma}=\begin{bmatrix}\pmb{\Sigma}_r & \pmb{0}\\ \pmb{0}&\pmb{0} \end{bmatrix}_{n\times n}
$$

&emsp;&emsp;因此，可以得到以下等式，

$$
\begin{split}
\pmb{A}&=\pmb{U}_r\pmb{\Sigma}_r\pmb{V}_r^H=\sum_{i=1}^r\sigma_i\pmb{u}_i\pmb{v}_i^H\\
\pmb{A}^H&=\pmb{V}_r\pmb{\Sigma}_r\pmb{U}_r^H=\sum_{i=1}^r\sigma_i\pmb{v}_i\pmb{u}_i^H\\
\end{split}
$$

###### **空间基构造的奇异值分解方法**

&emsp;&emsp;列空间基$\mathrm{Col}(\pmb{A})$为$r$个非零奇异值对应的左奇异向量$\pmb{u}_1,...,\pmb{u}_r$构成的空间。

$$
\begin{split}
\mathrm{Col}(\pmb{A})&=\mathrm{Range}(\pmb{A})=\{\pmb{y}\in\mathbb{C}^m|\pmb{y=Ax},\pmb{x}\in\mathbb{C}^n \}\\
&=\{\pmb{y}\in\mathbb{C}^m|\pmb{y}=\sum_{i=1}^r \pmb{u}_i(\sigma_i\pmb{v}_i^H\pmb{x}),\pmb{x}\in\mathbb{C}^n \}
\\
&=\mathrm{Span}(\pmb{u}_1,...,\pmb{u}_r)
\end{split}
$$

&emsp;&emsp;行空间基$\mathrm{Col}(\pmb{A})$为$r$个非零奇异值对应的右奇异向量$\pmb{v}_1,...,\pmb{v}_r$构成的空间，即

$$
\mathrm{Row}(\pmb{A})=\mathrm{Col}(\pmb{A}^H)=\mathrm{Span}(\pmb{v}_1,...,\pmb{v}_r)
$$

&emsp;&emsp;零空间基$\mathrm{Null}(\pmb{A})=\mathrm{Row}(\pmb{A})^\bot$，因此，零空间的基与行空间的基正交。由右奇异向量性质可知，

$$
\mathrm{Null}(\pmb{A})=\mathrm{Row}(\pmb{A})^\bot=\mathrm{Span}(\pmb{v}_{r+1},...,\pmb{v}_n)
$$

