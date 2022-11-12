### 向量空间


以向量为元素的集合称为**向量空间**。向量集合需要满足以下条件才能称之为向量空间：

1. 元素加法、标量乘法的**封闭性**。

   - 加法封闭，即 $\forall \pmb{v},\pmb{w}\in \pmb{V}$， 都有$\pmb{v+w }\in \pmb{V}$；

   - 数乘封闭，即$\forall \pmb{v}\in \pmb{V},c\in R$，都有$ c\pmb{v}\in \pmb{V}$。

2. 满足加法、数乘的以下**8个公理**。

| 公理 | 加法                                                         | 数乘                                               |
| ---- | ------------------------------------------------------------ | -------------------------------------------------- |
| 1    | $\pmb{u+(v+w)}=\pmb{(u+v)+w}$                          | $a(b\pmb{v})=(ab)\pmb{v}$                    |
| 2    | $\pmb{u+v}=\pmb{v+u}$                                  | $a(\pmb{u}+\pmb{v})=a\pmb{u}+a\pmb{v}$ |
| 3    | $\exists \pmb{0}\in V$,$\forall \pmb{v}\in V, \pmb{v}+\pmb{0})=\pmb{v}$ | $(a+b)\pmb{v}=a\pmb{v}+b\pmb{v}$          |
| 4    | $\forall \pmb{v}\in V, \exists -\pmb{v}\in V, \pmb{v}+(-\pmb{v})=0$ | $\forall 1\in R, 1\pmb{v}=\pmb{v}$           |

由于向量空间服从向量加法的交换律、结合律以及数乘的结合律、分配律，所以也称之为**线性空间**。




#### 子空间

1. 若$V,W$是两个向量空间，$W\subseteq V$，则称子集$W$是$V$的一个**子空间**，如果满足以下条件：

>1. $\forall \pmb{x,y}\in W$，都有$\pmb{x+y}\in W$成立；
>2. $\forall \pmb{x}\in W, c\in R$，都有$c\pmb{x}\in W$成立；
>3. $\pmb{0}\in W$；



2. 若$A$和$B$是向量空间$V$的两个子空间，则$A+B$和$A-B$也是$V$的一个子空间。



#### 线性映射

1. 若$V,W$是$\mathcal{R}^n$两个子空间，则

   
   $$
   T:V\rightarrow W
   $$
   

   称为子空间$V$到$W$的一个**映射**。它表示将子空间$V$中的每一个向量转换成子空间$W$的对应向量的规则。

   

2. 若$\forall \pmb{v}\in V, \pmb{w}\in W$，且映射$T$满足以下条件，

   - $T$满足叠加性：$T( \pmb{v+w})=T( \pmb{v})+T( \pmb{w})$
   - $T$满足齐次性：$T(c \pmb{v})=cT( \pmb{v})$

   则称映射$T$为**线性映射**。以上两个条件也可表示为**线性关系**，即：$T(c_1\pmb{v}+c_2\pmb{w})=c_1T(\pmb{v})+c_2T(\pmb{w})$。



#### 内积空间

在向量空间的基础上，增加向量乘法（内积）操作，即为内积空间。

##### 内积

若$\forall \pmb{x,y,z}\in V$和$\forall a,b\in R$，映射$\langle\cdot,\cdot\rangle: V\times V \rightarrow R$满足以下3个公理：

- 共轭对称性：$\langle\pmb{x},\pmb{y}\rangle=\langle\pmb{y},\pmb{x}\rangle$
- 第一变元线性：$\langle a\pmb{x}+b\pmb{y},\pmb{z}\rangle=a\langle \pmb{x},\pmb{z}\rangle+b\langle \pmb{y},\pmb{z}\rangle$
- 非负性：$\langle\pmb{x},\pmb{x}\rangle\ge 0$，且$\langle\pmb{x},\pmb{x}\rangle\ge 0\Leftrightarrow \pmb{x}=0$。

则称$\langle \pmb{x},\pmb{y}\rangle$为向量$\pmb{x}$与$\pmb{y}$的**内积**，$V$称为**内积向量空间**，简称**内积空间**。



特别地，如果向量内积$\langle\pmb{x},\pmb{y}\rangle$为0，则该对向量正交，即夹角为$90^。$，两向量互不干扰。

##### 加权内积

若$\forall G\succeq 0$，则向量$\pmb{x,y}$的加权内积为，


$$
\langle \pmb{x},\pmb{y}\rangle=\pmb{x}^\top G\pmb{y}=\sum_i\sum_j G_{ij}x_iy_j
$$




#### 赋范空间

在内积空间的基础上，新增向量长度、距离和领域等测度，即为赋范空间。

##### 范数

令$V$是一个向量空间，向量$\pmb{x}$的范数是一个实函数$p(\pmb{x}):\pmb{x}\rightarrow \mathbb{R}$且满足以下3个条件：

- 非负性： $p(\pmb{x})\ge 0$且$p(\pmb{x})=0\Leftrightarrow \pmb{x}=0$；
- 齐次性：$p(c\pmb{x})=|c|\cdot p(\pmb{x})$；
- 三角不等式：$p(\pmb{x}+\pmb{y})\le p(\pmb{x})+p(\pmb{y})$；

$V$也称之为赋范向量空间(normed vector space)。

##### 范数性质

- 极化恒等式：$\langle\pmb{x},\pmb{y}\rangle=\frac14\left(\lVert\pmb{x}+\pmb{y}\rVert^2-\lVert\pmb{x}-\pmb{y}\rVert^2\right)$
- 平行四边形法则：$\lVert\pmb{x}+\pmb{y}\rVert^2+\lVert\pmb{x}-\pmb{y}\rVert^2=2\left(\lVert\pmb{x}\rVert^2+\lVert\pmb{y}\rVert^2\right)$
- 三角不等式：$\lVert \pmb{x}+\pmb{y}\rVert\le\lVert\pmb{x}\rVert+\lVert \pmb{y}\rVert$
- Cauchy-Schwartz：$|\langle\pmb{x},\pmb{y}\rangle|\leq\lVert\pmb{x}\rVert\cdot\lVert \pmb{y}\rVert$



#### 完备向量空间

若对向量空间$V$中每一个Cauchy序列$\{\pmb{v}_n\}_{n=1}^\infty$，在空间$V$都存在一个元素$\pmb{v}$使之，


$$
\lim_{n\rightarrow\infty}\pmb{v}_n\rightarrow\pmb{v}
$$


成立，即$V$中的每一个Cauchy序列都收敛在向量空间$V$内，则称$V$为**完备向量空间**（元素收敛）；若有以下等式成立，


$$
\lim_{n\rightarrow \infty}\lVert\pmb{v}_n \rVert\rightarrow\lVert\pmb{v}\rVert
$$


则称该向量空间$V$为**相对于范数完备的向量空间**（范数收敛）。

##### Cauchy序列

对于任意序列$\{v_i\}_{i=1}^\infty$，若$\forall \epsilon >0$，总存在$\exists N$，当$m,n>N$时，$|x_n-x_m|<\epsilon$成立，则称序列$\{v_i\}$为一个Cauchy序列。换句话说，去掉有限个元素后，余下序列中的任意元素之间的距离都不超过给定的数$\epsilon$。





#### Banach空间

满足以下条件，


$$
\lim_{n\rightarrow \infty}\pmb{v}_n \rightarrow \pmb{v}
$$


的完备赋范向量空间，称为Banach空间。





#### Hilbert空间

满足以下条件，


$$
\lim_{n\rightarrow\infty}\lVert \pmb{v}_n\rVert\rightarrow\lVert \pmb{v}\rVert
$$


的相对于范数完备赋范向量空间，称为Hilbert空间。 显然，Hilbert空间一定是Banach空间。
