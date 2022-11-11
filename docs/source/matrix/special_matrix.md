### 特殊矩阵

#### Hermitian矩阵

&emsp;&emsp;**定义 (Hermitian矩阵).** 对于矩阵$A\in \mathbb{C}^{m\times m}$，若$a_{ij}=a^*_{ij}(\forall i,j | 1\le i<j \le m) $，即复矩阵的共轭转置$A^H=A$，则称矩阵$A$为Hermitian矩阵。

&emsp;&emsp;**定义 (共轭).** 两个实部相等，虚部相反的复数互为共轭复数。如$z=a+bi,\bar{z}=a-bi$。

简言之，矩阵对称元素共轭即为Hermitian矩阵，若矩阵为实矩阵，则该Hermitian矩阵为实对称矩阵。


| | Hermitain矩阵  |实对称矩阵   |
| :---:|:---:  |:---:    |
| 对称元素|   共轭    | 相等       |

&emsp;&emsp;**例1**：

```matlab
A=
   1.0000 + 0.0000i   2.0000 + 3.0000i   4.0000 - 1.0000i
   2.0000 - 3.0000i   2.0000 + 0.0000i   5.0000 + 6.0000i
   4.0000 + 1.0000i   5.0000 - 6.0000i   3.0000 + 0.0000i

A'=
   1.0000 + 0.0000i   2.0000 + 3.0000i   4.0000 - 1.0000i
   2.0000 - 3.0000i   2.0000 + 0.0000i   5.0000 + 6.0000i
   4.0000 + 1.0000i   5.0000 - 6.0000i   3.0000 + 0.0000i

 A == A'
   1   1   1
   1   1   1
   1   1   1

```



#### 酉矩阵

&emsp;&emsp;**定义（酉矩阵）** 复正方矩阵$U\in \mathbb{C}^{n\times n}$，若$UU^H=U^HU=I$，则该矩阵为酉矩阵。

&emsp;&emsp;**性质**

&emsp;&emsp;1. 若$A_{m\times m}$为酉矩阵，则$A^H,A^{-1},A^\top,A^i,A^*$均为酉矩阵，且$|\mathrm{det}(A)|=1,\mathrm{rank}(A)=m$。

&emsp;&emsp;2. 若$A_{m\times m},B_{m\times m}$为酉矩阵，则$AB$为酉矩阵。

&emsp;&emsp;3. 若$A_{m\times m}$为酉矩阵，则有$\forall B_{m\times n}$有，$\lVert AB\rVert_F=\lVert B\rVert_F$；对于$\forall B_{n\times m}$有，$\lVert BA\rVert_F=\lVert B\rVert_F$；对于$\forall x_{m\times 1}$有，$\lVert Ax\rVert_F=\lVert x\rVert_F$。

&emsp;&emsp;4. 若$A_{m\times m}$为酉矩阵，则$A$的所有行组成标准正交组；所有列组成标准正交组；$A$非奇异且$U^H=U^{-1}$。

#### 正交矩阵

&emsp;&emsp;**定义（正交矩阵）** 实矩阵$Q\in\mathbb{R}^{n\times n}$，若有$QQ^\top=Q^\top Q=I$，则该矩阵为正交矩阵。正交矩阵是酉矩阵的特例，即矩阵元素为实值的酉矩阵。



