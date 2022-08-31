### 矩阵运算

#### 直和

1. **定义**

所谓直和运算是指定义在任意矩阵$\mathbf{A}_{m\times m}$和$\mathbf{B}_{n\times n}$的运算，其规则如下，

$$
\mathbf{A}\oplus\mathbf{B}=\left[\begin{array}{cc}\mathbf{A}&\mathbf{0}_{m\times n}\\ \mathbf{0}_{n\times m}&\mathbf{B}\end{array}\right]
$$

记为$\mathbf{A}\oplus\mathbf{B}$。

2. **性质**

- $\mathbf{A}\oplus\mathbf{B}\neq\mathbf{B}\oplus\mathbf{A}$
- $c(\mathbf{A}\oplus\mathbf{B})=c\mathbf{A}\oplus c\mathbf{B}$
- $(\mathbf{A}\pm\mathbf{B})\oplus(\mathbf{C}\pm\mathbf{D})=(\mathbf{A}\oplus \mathbf{C})\pm(\mathbf{B}\oplus \mathbf{D})$
- $(\mathbf{A}\oplus\mathbf{B})(\mathbf{C}\oplus\mathbf{D})=\mathbf{AC}\oplus \mathbf{BD}$


#### Hadamard积

1. **定义**

Hadamard积是定义在任意同维度矩阵$\mathbf{A,B}\in\mathbb{R}^{m\times n}$的运算，记为$\mathbf{A}* \mathbf{B}$，也称为逐元素乘法，运算规则如下，

$$
[\mathbf{A}* \mathbf{B}]_{ij}=a_{ij}b_{ij}
$$

2. **性质**

- **正定性**. 若$\mathbf{A,B}$都是正定（半正定）矩阵，则$\mathbf{A}* \mathbf{B}$也正定（半正定）。
<font color="red">
- **迹相关**. $\mathrm{tr}[\mathbf{A}^\top(\mathbf{B}*\mathbf{C})]=\mathrm{tr}[(\mathbf{A}*\mathbf{B})^\top\mathbf{C}]=\mathrm{tr}[(\mathbf{A}^\top*\mathbf{B}^\top)\mathbf{C}]$
</font>
- $\mathbf{A}* \mathbf{B}=\mathbf{B}* \mathbf{A}$
- $\mathbf{A}* (\mathbf{B}* \mathbf{C})=(\mathbf{A}* \mathbf{B})* \mathbf{C}$
- $\mathbf{A}* (\mathbf{B}\pm \mathbf{C})=(\mathbf{A}* \mathbf{B})\pm (\mathbf{A}* \mathbf{C})$
- $(\mathbf{A}* \mathbf{B})^\top=\mathbf{A}^\top* \mathbf{B}^\top$
- $c(\mathbf{A}*\mathbf{B})=c\mathbf{A}* \mathbf{B}=\mathbf{A}* c\mathbf{B}$
- $(\mathbf{A}+\mathbf{B})*(\mathbf{C}+\mathbf{D})=\mathbf{A}* \mathbf{C}+\mathbf{A}* \mathbf{D}+\mathbf{B}* \mathbf{C}+\mathbf{B}* \mathbf{D}$
- $(\mathbf{A}\oplus\mathbf{B})*(\mathbf{C}\oplus\mathbf{D})=(\mathbf{A}*\mathbf{C})\oplus(\mathbf{B}*\mathbf{D})$
