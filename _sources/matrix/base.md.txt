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

