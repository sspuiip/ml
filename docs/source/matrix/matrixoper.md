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

- **迹相关**.<font color="red"> $\mathrm{tr}[\mathbf{A}^\top(\mathbf{B}*\mathbf{C})]=\mathrm{tr}[(\mathbf{A}*\mathbf{B})^\top\mathbf{C}]=\mathrm{tr}[(\mathbf{A}^\top*\mathbf{B}^\top)\mathbf{C}]$
</font>
- $\mathbf{A}* \mathbf{B}=\mathbf{B}* \mathbf{A}$
- $\mathbf{A}* (\mathbf{B}* \mathbf{C})=(\mathbf{A}* \mathbf{B})* \mathbf{C}$
- $\mathbf{A}* (\mathbf{B}\pm \mathbf{C})=(\mathbf{A}* \mathbf{B})\pm (\mathbf{A}* \mathbf{C})$
- $(\mathbf{A}* \mathbf{B})^\top=\mathbf{A}^\top* \mathbf{B}^\top$
- $c(\mathbf{A}*\mathbf{B})=c\mathbf{A}* \mathbf{B}=\mathbf{A}* c\mathbf{B}$
- $(\mathbf{A}+\mathbf{B})*(\mathbf{C}+\mathbf{D})=\mathbf{A}* \mathbf{C}+\mathbf{A}* \mathbf{D}+\mathbf{B}* \mathbf{C}+\mathbf{B}* \mathbf{D}$
- $(\mathbf{A}\oplus\mathbf{B})*(\mathbf{C}\oplus\mathbf{D})=(\mathbf{A}*\mathbf{C})\oplus(\mathbf{B}*\mathbf{D})$


#### Kronecker积

1. **定义**

对于任意矩阵$\mathbf{A}_{m\times n}$和$\mathbf{B}_{p\times q}$的Kronecker积定义为，

- 右积

$$
\mathbf{A}\otimes\mathbf{B}=[\mathbf{a}_1\mathbf{B},\cdots,\mathbf{a}_n\mathbf{B}]= \begin{bmatrix}a_{11}\mathbf{B}&a_{12}\mathbf{B}&\cdots&a_{1n}\mathbf{B}\\ a_{21}\mathbf{B}&a_{22}\mathbf{B}&\cdots&a_{2n}\mathbf{B}\\ \vdots&\vdots&\vdots&\vdots\\a_{m1}\mathbf{B}&a_{m2}\mathbf{B}&\cdots&a_{mn}\mathbf{B}\end{bmatrix}_{mp\times nq}
$$

- 左积

$$
[\mathbf{A}\otimes\mathbf{B}]_{\mathrm{left}}=[\mathbf{A}\mathbf{b}_1,\cdots,\mathbf{A}\mathbf{b}_n]= \begin{bmatrix}\mathbf{A}b_{11}&\mathbf{A}b_{12}&\cdots&\mathbf{A}b_{1q}\\ \mathbf{A}b_{21}&\mathbf{A}b_{22}&\cdots&\mathbf{A}b_{2q}\\ \vdots&\vdots&\vdots&\vdots\\\mathbf{A}b_{p1}&\mathbf{A}b_{p2}&\cdots&\mathbf{A}b_{pq}\\\end{bmatrix}_{mp\times nq}
$$

无论左积还是右积都是同一个映射$\mathbb{R}^{m\times n}\times\mathbb{R}^{p\times q}\rightarrow\mathbb{R}^{mp\times nq}$。可以看出$[\mathbf{A}\otimes\mathbf{B}]_{\mathrm{left}}=\mathbf{B}\otimes\mathbf{A}$，故默认采用右积。

当$n=q=1$时，

$$
\mathbf{a}\otimes \mathbf{b}=\begin{bmatrix}a_1b_1\\a_1b_2\\\vdots\\a_mb_p \end{bmatrix}_{mp\times 1}
$$

显然，向量外积$\mathbf{x}\mathbf{y}^\top$也可以写成Kronecker积的形式$\mathbf{x}\otimes\mathbf{y}^\top$，即

$$
\mathbf{x}\mathbf{y}^\top=\begin{bmatrix}x_1\mathbf{y}^\top\\x_2\mathbf{y}^\top\\\vdots \\x_m\mathbf{y}^\top\\ \end{bmatrix}_{m\times p}=\mathbf{x}\otimes\mathbf{y}^\top
$$


2. **性质**

- $\mathbf{A}\otimes\mathbf{B}\neq \mathbf{B}\otimes\mathbf{A}$
- $\mathbf{A}\otimes\mathbf{0}= \mathbf{0}\otimes\mathbf{A}=\mathbf{0}$
- $ab(\mathbf{A}\otimes\mathbf{B})=a\mathbf{A}\otimes b\mathbf{B}=b\mathbf{A}\otimes a\mathbf{B}$
- $\mathbf{I}_m\otimes\mathbf{I}_n=\mathbf{I}_{mn}$
- $(\mathbf{AB})\otimes(\mathbf{CD})=(\mathbf{A}\otimes\mathbf{C})(\mathbf{B}\otimes\mathbf{D})$
- $\mathbf{A}\otimes(\mathbf{B}\pm\mathbf{C})=(\mathbf{A}\otimes\mathbf{B})\pm(\mathbf{A}\otimes\mathbf{C})$
- $(\mathbf{A}\otimes\mathbf{B})^\top=\mathbf{A}^\top\otimes\mathbf{B}^\top$
- $(\mathbf{A}\otimes\mathbf{B})^{-1}=\mathbf{A}^{-1}\otimes\mathbf{B}^{-1}$
- $\mathrm{rank}(\mathbf{A}\otimes\mathbf{B})=\mathrm{rank}(\mathbf{A})\mathrm{rank}(\mathbf{B})$
- $|\mathbf{A}_{n\times n}\otimes\mathbf{B}_{m\times m}|=|\mathbf{A}|^m|\mathbf{B}|^n$
- $\mathrm{tr}(\mathbf{A}\otimes\mathbf{B})=\mathrm{tr}(\mathbf{A})\mathrm{tr}(\mathbf{B})$
- $(\mathbf{A}+\mathbf{B})\otimes(\mathbf{C}+\mathbf{D})=\mathbf{A}\otimes\mathbf{C}+\mathbf{A}\otimes\mathbf{D}+\mathbf{B}\otimes\mathbf{C}+\mathbf{B}\otimes\mathbf{D}$
- $(\mathbf{A}\otimes\mathbf{B})\otimes\mathbf{C}=\mathbf{A}\otimes(\mathbf{B}\otimes\mathbf{C})$
- $(\mathbf{A}\otimes\mathbf{B})\otimes(\mathbf{C}\otimes\mathbf{D})=\mathbf{A}\otimes\mathbf{B}\otimes\mathbf{C}\otimes\mathbf{D}$
- $(\mathbf{A}\otimes\mathbf{B})(\mathbf{C}\otimes\mathbf{D})(\mathbf{E}\otimes\mathbf{F})=(\mathbf{ACE})\otimes(\mathbf{BDF})$，特别地，$\mathbf{A}\otimes\mathbf{D}=(\mathbf{AI}_p)\otimes(\mathbf{I}_q\mathbf{D})=(\mathbf{A}\otimes\mathbf{I}_q)(\mathbf{I}_p\otimes\mathbf{D})$
- $\exp(\mathbf{A}\otimes\mathbf{B})=\exp(\mathbf{A})\otimes\exp(\mathbf{B})$

#### 向量化

1. **定义**

矩阵的向量化指的是将矩阵按列序排成一个向量的操作。如，

$$
\mathbf{A}=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22} \end{bmatrix}
$$

列向量化后，结果为，

$$
\mathrm{vec}(\mathbf{A})=\begin{bmatrix}a_{11}\\a_{21}\\a_{12}\\a_{22} \end{bmatrix}
$$

2. **性质**

- $\mathrm{vec}(\mathbf{A}^\top)=\mathbf{K}_{mn}\mathrm{vec}(\mathbf{A})$
- $\mathrm{vec}(\mathbf{A}+\mathbf{B})=\mathrm{vec}(\mathbf{A})+\mathrm{vec}(\mathbf{B})$
- $\mathrm{tr}(\mathbf{A}^\top\mathbf{B})=(\mathrm{vec}\mathbf{A})^\top\mathrm{vec}(\mathbf{B})$
- $\mathrm{tr}(\mathbf{A}\mathbf{B}\mathbf{C})=[\mathrm{vec}(\mathbf{A}^\top)]^\top(\mathbf{I}_p\otimes\mathbf{B})\mathrm{vec}\mathbf{C}=[\mathrm{vec}(\mathbf{AB})^\top]^\top\mathrm{vec}(\mathbf{C})$

- 多矩阵相乘

 $$
 \mathrm{vec}(\mathbf{A}_{m\times p}\mathbf{B}_{p\times q}\mathbf{C}_{q\times n})=(\mathbf{C}^\top\otimes \mathbf{A})\mathrm{vec}(\mathbf{B})=(\mathbf{I}_q\otimes\mathbf{AB})\mathrm{vec}(\mathbf{C})=(\mathbf{C}^\top\mathbf{B}^\top\otimes\mathbf{I}_m)\mathrm{vec}(\mathbf{A})
 $$

 - <font color="red">两矩阵乘积向量化</font>

 $$
 \mathrm{vec}(\mathbf{A}\mathbf{C})=(\mathbf{I}_p\otimes \mathbf{A})\mathrm{vec}(\mathbf{C})=(\mathbf{C}^\top\otimes\mathbf{I}_m)\mathrm{vec}(\mathbf{A})
 $$

 3. **例**
 
 向量化求解矩阵方程$\mathbf{AX}+\mathbf{XB}=\mathbf{Y}$的解$\hat{\mathbf{X}}$，其中所有矩阵均为$n\times n$阶矩阵。

 对等式左右同时向量化，由向量化公式可知，

 $$
(\mathbf{I}_n\otimes\mathbf{A}+\mathbf{B}^\top\otimes\mathbf{I}_n)\mathrm{vec}(\mathbf{X})=\mathrm{vec}(\mathbf{Y})
 $$

 则，

 $$
\mathrm{vec}(\mathbf{X})=(\mathbf{I}_n\otimes\mathbf{A}+\mathbf{B}^\top\otimes\mathbf{I}_n)^{\dagger}\mathrm{vec}(\mathbf{Y})
 $$

 最后再矩阵化即为$\hat{\mathbf{X}}$的值。