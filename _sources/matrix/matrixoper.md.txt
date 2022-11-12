### 矩阵运算

#### 直和

1. **定义**

所谓直和运算是指定义在任意矩阵$\pmb{A}_{m\times m}$和$\pmb{B}_{n\times n}$的运算，其规则如下，

$$
\pmb{A}\oplus\pmb{B}=\left[\begin{array}{cc}\pmb{A}&\pmb{0}_{m\times n}\\ \pmb{0}_{n\times m}&\pmb{B}\end{array}\right]
$$

记为$\pmb{A}\oplus\pmb{B}$。

2. **性质**

- $\pmb{A}\oplus\pmb{B}\neq\pmb{B}\oplus\pmb{A}$
- $c(\pmb{A}\oplus\pmb{B})=c\pmb{A}\oplus c\pmb{B}$
- $(\pmb{A}\pm\pmb{B})\oplus(\pmb{C}\pm\pmb{D})=(\pmb{A}\oplus \pmb{C})\pm(\pmb{B}\oplus \pmb{D})$
- $(\pmb{A}\oplus\pmb{B})(\pmb{C}\oplus\pmb{D})=\pmb{AC}\oplus \pmb{BD}$


#### Hadamard积

1. **定义**

Hadamard积是定义在任意同维度矩阵$\pmb{A,B}\in\mathbb{R}^{m\times n}$的运算，记为$\pmb{A}* \pmb{B}$，也称为逐元素乘法，运算规则如下，

$$
[\pmb{A}* \pmb{B}]_{ij}=a_{ij}b_{ij}
$$

2. **性质**

- **正定性**. 若$\pmb{A,B}$都是正定（半正定）矩阵，则$\pmb{A}* \pmb{B}$也正定（半正定）。

- **迹相关**.<font color="red"> $\mathrm{tr}[\pmb{A}^\top(\pmb{B}*\pmb{C})]=\mathrm{tr}[(\pmb{A}*\pmb{B})^\top\pmb{C}]=\mathrm{tr}[(\pmb{A}^\top*\pmb{B}^\top)\pmb{C}]$
</font>
- $\pmb{A}* \pmb{B}=\pmb{B}* \pmb{A}$
- $\pmb{A}* (\pmb{B}* \pmb{C})=(\pmb{A}* \pmb{B})* \pmb{C}$
- $\pmb{A}* (\pmb{B}\pm \pmb{C})=(\pmb{A}* \pmb{B})\pm (\pmb{A}* \pmb{C})$
- $(\pmb{A}* \pmb{B})^\top=\pmb{A}^\top* \pmb{B}^\top$
- $c(\pmb{A}*\pmb{B})=c\pmb{A}* \pmb{B}=\pmb{A}* c\pmb{B}$
- $(\pmb{A}+\pmb{B})*(\pmb{C}+\pmb{D})=\pmb{A}* \pmb{C}+\pmb{A}* \pmb{D}+\pmb{B}* \pmb{C}+\pmb{B}* \pmb{D}$
- $(\pmb{A}\oplus\pmb{B})*(\pmb{C}\oplus\pmb{D})=(\pmb{A}*\pmb{C})\oplus(\pmb{B}*\pmb{D})$


#### Kronecker积

1. **定义**

对于任意矩阵$\pmb{A}_{m\times n}$和$\pmb{B}_{p\times q}$的Kronecker积定义为，

- 右积

$$
\pmb{A}\otimes\pmb{B}=[\pmb{a}_1\pmb{B},\cdots,\pmb{a}_n\pmb{B}]= \begin{bmatrix}a_{11}\pmb{B}&a_{12}\pmb{B}&\cdots&a_{1n}\pmb{B}\\ a_{21}\pmb{B}&a_{22}\pmb{B}&\cdots&a_{2n}\pmb{B}\\ \vdots&\vdots&\vdots&\vdots\\a_{m1}\pmb{B}&a_{m2}\pmb{B}&\cdots&a_{mn}\pmb{B}\end{bmatrix}_{mp\times nq}
$$

- 左积

$$
[\pmb{A}\otimes\pmb{B}]_{\mathrm{left}}=[\pmb{A}\pmb{b}_1,\cdots,\pmb{A}\pmb{b}_n]= \begin{bmatrix}\pmb{A}b_{11}&\pmb{A}b_{12}&\cdots&\pmb{A}b_{1q}\\ \pmb{A}b_{21}&\pmb{A}b_{22}&\cdots&\pmb{A}b_{2q}\\ \vdots&\vdots&\vdots&\vdots\\\pmb{A}b_{p1}&\pmb{A}b_{p2}&\cdots&\pmb{A}b_{pq}\\\end{bmatrix}_{mp\times nq}
$$

无论左积还是右积都是同一个映射$\mathbb{R}^{m\times n}\times\mathbb{R}^{p\times q}\rightarrow\mathbb{R}^{mp\times nq}$。可以看出$[\pmb{A}\otimes\pmb{B}]_{\mathrm{left}}=\pmb{B}\otimes\pmb{A}$，故默认采用右积。

当$n=q=1$时，

$$
\pmb{a}\otimes \pmb{b}=\begin{bmatrix}a_1b_1\\a_1b_2\\\vdots\\a_mb_p \end{bmatrix}_{mp\times 1}
$$

显然，向量外积$\pmb{x}\pmb{y}^\top$也可以写成Kronecker积的形式$\pmb{x}\otimes\pmb{y}^\top$，即

$$
\pmb{x}\pmb{y}^\top=\begin{bmatrix}x_1\pmb{y}^\top\\x_2\pmb{y}^\top\\\vdots \\x_m\pmb{y}^\top\\ \end{bmatrix}_{m\times p}=\pmb{x}\otimes\pmb{y}^\top
$$


2. **性质**

- $\pmb{A}\otimes\pmb{B}\neq \pmb{B}\otimes\pmb{A}$
- $\pmb{A}\otimes\pmb{0}= \pmb{0}\otimes\pmb{A}=\pmb{0}$
- $ab(\pmb{A}\otimes\pmb{B})=a\pmb{A}\otimes b\pmb{B}=b\pmb{A}\otimes a\pmb{B}$
- $\pmb{I}_m\otimes\pmb{I}_n=\pmb{I}_{mn}$
- $(\pmb{AB})\otimes(\pmb{CD})=(\pmb{A}\otimes\pmb{C})(\pmb{B}\otimes\pmb{D})$
- $\pmb{A}\otimes(\pmb{B}\pm\pmb{C})=(\pmb{A}\otimes\pmb{B})\pm(\pmb{A}\otimes\pmb{C})$
- $(\pmb{A}\otimes\pmb{B})^\top=\pmb{A}^\top\otimes\pmb{B}^\top$
- $(\pmb{A}\otimes\pmb{B})^{-1}=\pmb{A}^{-1}\otimes\pmb{B}^{-1}$
- $\mathrm{rank}(\pmb{A}\otimes\pmb{B})=\mathrm{rank}(\pmb{A})\mathrm{rank}(\pmb{B})$
- $|\pmb{A}_{n\times n}\otimes\pmb{B}_{m\times m}|=|\pmb{A}|^m|\pmb{B}|^n$
- $\mathrm{tr}(\pmb{A}\otimes\pmb{B})=\mathrm{tr}(\pmb{A})\mathrm{tr}(\pmb{B})$
- $(\pmb{A}+\pmb{B})\otimes(\pmb{C}+\pmb{D})=\pmb{A}\otimes\pmb{C}+\pmb{A}\otimes\pmb{D}+\pmb{B}\otimes\pmb{C}+\pmb{B}\otimes\pmb{D}$
- $(\pmb{A}\otimes\pmb{B})\otimes\pmb{C}=\pmb{A}\otimes(\pmb{B}\otimes\pmb{C})$
- $(\pmb{A}\otimes\pmb{B})\otimes(\pmb{C}\otimes\pmb{D})=\pmb{A}\otimes\pmb{B}\otimes\pmb{C}\otimes\pmb{D}$
- $(\pmb{A}\otimes\pmb{B})(\pmb{C}\otimes\pmb{D})(\pmb{E}\otimes\pmb{F})=(\pmb{ACE})\otimes(\pmb{BDF})$，特别地，$\pmb{A}\otimes\pmb{D}=(\pmb{AI}_p)\otimes(\pmb{I}_q\pmb{D})=(\pmb{A}\otimes\pmb{I}_q)(\pmb{I}_p\otimes\pmb{D})$
- $\exp(\pmb{A}\otimes\pmb{B})=\exp(\pmb{A})\otimes\exp(\pmb{B})$

#### 向量化

1. **定义**

矩阵的向量化指的是将矩阵按列序排成一个向量的操作。如，

$$
\pmb{A}=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22} \end{bmatrix}
$$

列向量化后，结果为，

$$
\mathrm{vec}(\pmb{A})=\begin{bmatrix}a_{11}\\a_{21}\\a_{12}\\a_{22} \end{bmatrix}
$$

2. **性质**

- $\mathrm{vec}(\pmb{A}^\top)=\pmb{K}_{mn}\mathrm{vec}(\pmb{A})$
- $\mathrm{vec}(\pmb{A}+\pmb{B})=\mathrm{vec}(\pmb{A})+\mathrm{vec}(\pmb{B})$
- $\mathrm{tr}(\pmb{A}^\top\pmb{B})=(\mathrm{vec}\pmb{A})^\top\mathrm{vec}(\pmb{B})$
- $\mathrm{tr}(\pmb{A}\pmb{B}\pmb{C})=[\mathrm{vec}(\pmb{A}^\top)]^\top(\pmb{I}_p\otimes\pmb{B})\mathrm{vec}\pmb{C}=[\mathrm{vec}(\pmb{AB})^\top]^\top\mathrm{vec}(\pmb{C})$

- 多矩阵相乘

 $$
 \mathrm{vec}(\pmb{A}_{m\times p}\pmb{B}_{p\times q}\pmb{C}_{q\times n})=(\pmb{C}^\top\otimes \pmb{A})\mathrm{vec}(\pmb{B})=(\pmb{I}_q\otimes\pmb{AB})\mathrm{vec}(\pmb{C})=(\pmb{C}^\top\pmb{B}^\top\otimes\pmb{I}_m)\mathrm{vec}(\pmb{A})
 $$

 - <font color="red">两矩阵乘积向量化</font>

 $$
 \mathrm{vec}(\pmb{A}\pmb{C})=(\pmb{I}_p\otimes \pmb{A})\mathrm{vec}(\pmb{C})=(\pmb{C}^\top\otimes\pmb{I}_m)\mathrm{vec}(\pmb{A})
 $$

 3. **例**
 
 向量化求解矩阵方程$\pmb{AX}+\pmb{XB}=\pmb{Y}$的解$\hat{\pmb{X}}$，其中所有矩阵均为$n\times n$阶矩阵。

 对等式左右同时向量化，由向量化公式可知，

 $$
(\pmb{I}_n\otimes\pmb{A}+\pmb{B}^\top\otimes\pmb{I}_n)\mathrm{vec}(\pmb{X})=\mathrm{vec}(\pmb{Y})
 $$

 则，

 $$
\mathrm{vec}(\pmb{X})=(\pmb{I}_n\otimes\pmb{A}+\pmb{B}^\top\otimes\pmb{I}_n)^{\dagger}\mathrm{vec}(\pmb{Y})
 $$

 最后再矩阵化即为$\hat{\pmb{X}}$的值。