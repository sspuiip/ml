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


