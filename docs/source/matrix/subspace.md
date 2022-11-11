### 子空间分析

#### 什么是子空间

##### 概念

&emsp;&emsp;**定义**  假设$S=\{\mathbf{u}_1,...,\mathbf{u}_m\}$是向量空间$V$的向量子集，则$\mathbf{u}_1,...,\mathbf{u}_m$的所有线性组合的集合$W$称为由$\mathbf{u}_1,...,\mathbf{u}_m$张成的**子空间**，即

$$
W=\mathrm{span}(\mathbf{u}_1,...,\mathbf{u}_m)=\{\mathbf{u}|\mathbf{u}=a_1\mathbf{u}_1+...+a_m\mathbf{u}_m\}
$$

其中$W$称为子空间的**张成集**，每个向量称为$W$的**生成元**。

##### 正交补

&emsp;&emsp;子空间$S_1,...,S_n$的交

$$
S=S_1\cap S_2\cap ... \cap S_n
$$

是所有子空间共有的向量组成的集合。

&emsp;&emsp;**定义** 若$S=\{\mathbf{0}\}$，则称子空间$S_1,...,S_n$为**无交连**。

&emsp;&emsp;**定义** 无交连的子空间并集$S=S_1\cup S_2\cup ... \cup S_n$称为**子空间的直和**，记为，

$$
S=S_1 \oplus S_2 \oplus ... \oplus S_n
$$

&emsp;&emsp;**定义** 子空间$S$正交的所有向量的集合组成一个向量子空间，称为$S$的**正交补**空间，记为$S^\bot$。

&emsp;&emsp;1.  正交连的两个子空间不一定正交，但正交的两个子空间必定是无交连。

&emsp;&emsp;2.  正交补空间是比正交空间更严格的概念。

&emsp;&emsp;**定义** 若下式成立，则称子空间为相对于$\mathbf{A}$不变的，

$$
\forall\mathbf{x}\in S \quad\Rightarrow\quad \mathbf{Ax}\in S
$$

##### 正交投影

&emsp;&emsp;假设有两个子空间$S,H$，如果有线性变换矩阵$\mathbf{P}$，将$\mathbf{x}\in\mathbb{R}^n$映射为子空间$S$的向量$\mathbf{x}_1$，则这种线性变换称为沿着$H$的方向到$S$的**投影算子**，记为$\mathbf{P}_{S|H}$。特别地，若$S,H$是正交补，则$\mathbf{P}_{S|S^\bot}$是将$\mathbf{x}$沿着与子空间$S$垂直方向的投影，称为子空间的**正交投影**，记为$\mathbf{P}_S$。


&emsp;&emsp;假设两个子空间$S,H$其交集为空$\{\mathbf{0}\}$，其直和空间为$V=S\oplus H$，则直和空间向量可分解为，

$$
\mathbf{x}=\mathbf{x}_s+\mathbf{x}_h
$$

其中，$\mathbf{x}_s\in S, \mathbf{x}_h\in H$。由于$S\cap H=\{\mathbf{0}\}$，则该分解唯一。定义一个沿着子空间$H$到子空间$S$的投影算子$\mathbf{P}_{S|H}$，即，

$$
\mathbf{Px}=\mathbf{x}_s
$$

代入直和分解向量，则有，

$$
\mathbf{x}_h=(\mathbf{I}-\mathbf{P})\mathbf{x}
$$

即，$\mathbf{I}-\mathbf{P}$是沿着子空间$S$到子空间$H$的投影算子。因此，可知，

$$
\mathbf{x}=\mathbf{Px}+(\mathbf{I}-\mathbf{P})\mathbf{x}
$$

以及，子空间的内部向量投影就是其本身，即$\mathbf{Px}_1=\mathbf{x}_1$,从而有，

$$
\mathbf{P}^2\mathbf{x}_1=\mathbf{Px}_1=\mathbf{x}_1
$$

也就是幂等性，

$$
\mathbf{P}^2=\mathbf{P}
$$

&emsp;&emsp;**定义** 若$Range(\mathbf{P})=S,\mathbf{P}^2=\mathbf{P}$和$\mathbf{P}^\top=\mathbf{P}$，则矩阵$\mathbf{P}$称为到子空间$S$的**正交投影**。

&emsp;&emsp;观察可知，线性变换矩阵，

$$
\mathbf{P}_S=\mathbf{A}(\mathbf{A}^\mathrm{H}\mathbf{A})^{-1}\mathbf{A}^\mathrm{H}
$$

满足正交投影算子定义的幂等性和Hermitian性，且$\mathbf{P}_S\mathbf{A}=\mathbf{A}$即满足$Range(\mathbf{P})=S$$=Span(\mathbf{A})$。因此该矩阵是到由$\mathbf{A}$的列向量生成的子空间$S$上的正交投影算子。

