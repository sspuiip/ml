### 矩阵微分

矩阵微分是多变量函数微分的推广。首先注意区分以下标记。

| **标记**  |  **含义** | **变量**  |  **映射**   |
|:--|:--|:--| :-- |
| $\mathbf{x}=[x_1,\cdots,x_m]^\top\in\mathbb{R}^m$  |实向量变量    |     |    |
|  $\mathbf{X}=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^\top\in\mathbb{R}^{m\times n}$ |  矩阵变量  |     |    |
|$f(\mathbf{x})\in \mathbb{R}$ | 实标量函数   |  $\mathbf{x}\in\mathbb{R}^m$   | $f:\mathbb{R}^m\rightarrow\mathbb{R}$   |
| $f(\mathbf{X})\in \mathbb{R}$  | 实标量函数  | $\mathbf{X}\in\mathbb{R}^{m\times n}$    |  $f:\mathbb{R}^{m\times n}\rightarrow\mathbb{R}$  |
| $\mathbf{f}(\mathbf{x})\in \mathbb{R}^p$  | $p$维列向量函数   |  $\mathbf{x}\in\mathbb{R}^m$   | $f:\mathbb{R}^m\rightarrow\mathbb{R}^p$   |
|$\mathbf{f}(\mathbf{X})\in \mathbb{R}^p$   | $p$维列向量函数   | $\mathbf{X}\in\mathbb{R}^{m\times n}$    | $f:\mathbb{R}^{m\times n}\rightarrow\mathbb{R}^p$   |
| $\mathbf{F}(\mathbf{x})\in \mathbb{R}^{p\times q}$  | $p\times q$维矩阵函数   |$\mathbf{x}\in\mathbb{R}^m$     |$f:\mathbb{R}^m\rightarrow\mathbb{R}^{p\times q}$    |
| $\mathbf{F}(\mathbf{X})\in \mathbb{R}^{p\times q}$  |${p\times q}$维矩阵函数    | $\mathbf{X}\in\mathbb{R}^{m\times n}$    | $f:\mathbb{R}^{m\times n}\rightarrow\mathbb{R}^{p\times q}$   |

#### Jacobian矩阵

1. **向量偏导算子（分子布局）**。
$1\times m$行向量$\mathbf{x}^\top$**偏导算子**记为，

$$
\mathbf{D}_{\mathbf{x}}\triangleq\frac{\partial}{\partial\mathbf{x}^\top}=\left[\frac{\partial}{\partial x_1},\cdots,\frac{\partial}{\partial x_m} \right]
$$

2. **标量函数偏导向量**。
标量函数$f(\mathbf{x})$在$\mathbf{x}$的偏导向量由如下$1\times m$行向量给出，

$$
\mathbf{D}_{\mathbf{x}}f(\mathbf{x})=\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}=\left[\frac{\partial f(\mathbf{x})}{\partial x_1},\cdots,\frac{\partial f(\mathbf{x})}{\partial x_m} \right]
$$

3. 标量函数$f(\mathbf{X})$的变元为矩阵$\mathbf{X}\in \mathbb{R}^{m\times n}$时有，

    - 定义1 (**Jacobian矩阵**)

    $$
    \mathbf{D}_{\mathbf{X}}f(\mathbf{X})=\frac{\partial f(\mathbf{X})}{\partial \mathbf{X}^\top}=\left[\begin{array}{ccc} \frac{\partial f(\mathbf{X})}{\partial x_{11}} &\cdots & \frac{\partial f(\mathbf{X})}{\partial x_{m1}}\\ \vdots &\ddots & \vdots \\ \frac{\partial f(\mathbf{X})}{\partial x_{1n}} &\cdots & \frac{\partial f(\mathbf{X})}{\partial x_{mn}}\end{array} \right]\in \mathbb{R}^{n\times m}
    $$

    - 定义2 (**行向量偏导**)

    $$
    \mathbf{D}_{\mathrm{vec}\mathbf{X}}f(\mathbf{X})=\frac{\partial f(\mathbf{X})}{\partial \mathrm{vec}^\top (\mathbf{X})}=\left[\frac{\partial f(\mathbf{X})}{\partial x_{11}},\cdots,\frac{\partial f(\mathbf{X})}{\partial x_{m1}},\cdots,\frac{\partial f(\mathbf{X})}{\partial x_{1n}},\cdots,\frac{\partial f(\mathbf{X})}{\partial x_{mn}}    \right]
    $$

两者之间的关系为，

$$
\mathbf{D}_{\mathrm{vec}\mathbf{X}}f(\mathbf{X})=\left[\mathrm{vec}\left(\mathbf{D}_{\mathbf{X}}^\top f(\mathbf{X})\right)\right]^\top
$$

即标量函数的行向量偏导$\mathbf{D}_{\mathrm{vec}\mathbf{X}}f(\mathbf{X})$等于标量函数$f(\mathbf{X})$关于矩阵变元$\mathbf{X}$的Jacobian矩阵。

#### 梯度矩阵

**梯度算子(分母布局)** 指的是以列向量形式$\mathbf{x}=[x_1,...,x_m]^\top$定义的偏导算子，记为$\nabla_{\mathbf{x}}$，定义为

$$
\nabla_{\mathbf{x}}\triangleq\frac{\partial}{\partial \mathbf{x}}=\left[\frac{\partial}{\partial x_1},\cdots,\frac{\partial}{\partial x_m} \right]^\top
$$

1. 标量函数$f(\mathbf{x})$的**梯度向量**$\nabla_{\mathbf{x}}f(\mathbf{x})$为1个与$\mathbf{x}$同维度的列向量，即

$$
\nabla_{\mathbf{x}}f(\mathbf{x})\triangleq\left[\frac{\partial f(\mathbf{x})}{\partial x_1},\cdots,\frac{\partial f(\mathbf{x})}{\partial x_m}\right]^\top=\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}\neq \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}=\mathbf{D}_{\mathbf{x}}f(\mathbf{x})
$$

2. 当标量函数的变元为矩阵$\mathbf{X}\in \mathbb{R}^{m\times n}$,并列向理化后，函数$f(\mathbf{X})$关于矩阵$\mathbf{X}$的**梯度向量**为，

$$
\nabla_{\mathrm{vec}\mathbf{X}}f(\mathbf{X})=\frac{\partial f(\mathbf{X})}{\partial \mathrm{vec}\mathbf{X}}=\left[\frac{\partial  f(\mathbf{X})}{\partial x_{11}},\cdots,\frac{\partial  f(\mathbf{X})}{\partial x_{m1}},\cdots,\frac{\partial  f(\mathbf{X})}{\partial x_{1n}},\cdots,\frac{\partial  f(\mathbf{X})}{\partial x_{mn}}\right]^\top
$$

3. 也可以直接定义**梯度矩阵**

$$
\nabla_{\mathbf{X}}f(\mathbf{X})=\left[\begin{array}{ccc} \frac{\partial f(\mathbf{X})}{\partial x_{11}} &\cdots&\frac{\partial f(\mathbf{X})}{\partial x_{1n}}\\\vdots&\ddots&\vdots\\ \frac{\partial f(\mathbf{X})}{\partial x_{m1}}&\cdots &\frac{\partial f(\mathbf{X})}{\partial x_{mn}} \end{array} \right]=\frac{\partial f(\mathbf{X})}{\partial \mathbf{x}}
$$

可以看出，梯度矩阵$\nabla_{\mathbf{x}}f(\mathbf{X})$是梯度向量$\nabla_{\mathrm{vec}\mathbf{X}}f(\mathbf{X})$的矩阵化，以及以下关系，

$$
\nabla_{\mathbf{X}}f(\mathbf{X})=\mathbf{D}_{\mathbf{X}}^\top f(\mathbf{X})
$$

即，标量函数的梯度矩阵等价于Jacobian矩阵的转置。

#### 偏导和梯度计算

##### 基本规则

假设$\mathbf{X}\in \mathbb{R}^{m\times n}$，$f(\mathbf{X})$为一标量函数，则有以下偏导计算规则：

1. 若$f(\mathbf{X})=c$为常数，则梯度$\frac{\partial c}{\partial \mathbf{X}}=\mathbf{0}_{m\times n}$。

2. **线性规则** 若$f(\mathbf{X}),g(\mathbf{X})$分别为矩阵$\mathbf{X}$的实值函数,$c_1,c_2$为实常数，则有以下等式成立，

$$
\frac{\partial [c_1f(\mathbf{X})+c_2g(\mathbf{X})]}{\partial \mathbf{X}}=c_1\frac{\partial f(\mathbf{X})}{\partial \mathbf{X}}+c_2\frac{\partial g(\mathbf{X})}{\partial \mathbf{X}}
$$

3. **乘法规则** 若$f(\mathbf{X}),g(\mathbf{X}),h(\mathbf{X})$分别为矩阵$\mathbf{X}$的实值函数，则有以下等式成立，

$$
\frac{\partial [f(\mathbf{X})g(\mathbf{X})]}{\partial \mathbf{X}}=g(\mathbf{X})\frac{\partial f(\mathbf{X})}{\partial \mathbf{X}}+f(\mathbf{X})\frac{\partial g(\mathbf{X})}{\partial \mathbf{X}}
$$

和，

$$
\frac{\partial [f(\mathbf{X})g(\mathbf{X})h(\mathbf{X})]}{\partial \mathbf{X}}=g(\mathbf{X})h(\mathbf{X})\frac{\partial f(\mathbf{X})}{\partial \mathbf{X}}+
f(\mathbf{X})h(\mathbf{X})\frac{\partial g(\mathbf{X})}{\partial \mathbf{X}}+
f(\mathbf{X})g(\mathbf{X})\frac{\partial h(\mathbf{X})}{\partial \mathbf{X}}
$$

4. **商规则** 若$(\mathbf{X})\neq 0$，则有，

$$
\frac{\partial [f(\mathbf{X})/g(\mathbf{X})]}{\partial \mathbf{X}}=\frac{1}{g^2(\mathbf{X})}\left[ g(\mathbf{X})\frac{\partial f(\mathbf{X})}{\partial\mathbf{X}}-f(\mathbf{X})\frac{\partial g(\mathbf{X})}{\partial\mathbf{X}}\right]
$$

5. **链式规则** 假设$y=f(\mathbf{X})$和$g(y)$分别是以矩阵$\mathbf{X}$和标量$y$为变元的实值函数，则

$$
\frac{\partial g[f(\mathbf{X})]}{\partial \mathbf{X}}=\frac{dg(y)}{dy}\frac{\partial f(\mathbf{X})}{\partial\mathbf{X}}
$$

推广，记$g(\mathbf{F}(\mathbf{X}))=g(\mathbf{F})$，其中，$\mathbf{F}=[f_{kl}]\in \mathbb{R}^{p\times q}, \mathbf{X}\in \mathbb{R}^{m\times n}$，则链式法则为，

$$
\left[\frac{\partial g(\mathbf{F})}{\partial \mathbf{X}}\right]_{ij}=\frac{\partial g(\mathbf{F})}{\partial x_{ij}}=\sum_{k=1}^p\sum_{l=1}^q \frac{\partial g(\mathbf{F})}{\partial f_{kl}}\frac{\partial f_{kl}}{\partial x_{ij}}
$$

###### 独立性假设

假定实值函数的向量变元$\mathbf{x}=[x_i]_{i=1}^m\in\mathbb{R}^m$或者矩阵变元$\mathbf{X}_{ij}\in\mathbb{R}^{m\times n}$本身无任何特殊结构，也就是向量或矩阵变元的元素之间是相互独立的，即

$$
\frac{\partial x_i}{\partial x_j}=\delta_{ij}=\left\{\begin{array}{ll}1,&i=j\\ 0,&i\neq j \end{array} \right.
$$

以及，

$$
\frac{\partial x_{kl}}{\partial x_{ij}}=\delta_{ki}\delta_{lj}=\left\{\begin{array}{ll}1,&k=i \wedge l=j\\ 0,&others \end{array} \right.
$$

###### 案例

1. 求实值函数$f(\mathbf{x})=\mathbf{x}^\top\mathbf{A}\mathbf{x}$的梯度向量与Jacobian矩阵。由于$\mathbf{x}^\top\mathbf{A}\mathbf{x}=\sum_{k=1}^n\sum_{l=1}^n a_{kl}x_kx_l$，则根据$\mathbf{D}_{\mathbf{x}}f(\mathbf{x})=\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}$可知第$i$分量为，

$$
\left[\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}\right]_i=\frac{\partial f(\mathbf{x})}{\partial x_i}=\sum_{k=1}^n\sum_{l=1}^n\frac{\partial a_{kl}x_kx_l}{\partial x_i}=\sum_{l=1}^na_{il}x_l+\sum_{i=1}^na_{ki}x_k=[\mathbf{x}^\top\mathbf{A}^\top_i]+[\mathbf{x}^\top\mathbf{A}_i]
$$

其中，$\mathbf{A}_i$为矩阵$\mathbf{A}$的第$i$列。可知，$\mathbf{D}f(\mathbf{x})=\mathbf{x}^\top(\mathbf{A}+\mathbf{A}^\top)$(根据公式可知**标量函数对行向量求偏导的计算结果为一行向量**)， 同理可知梯度向量$\nabla_\mathbf{x}f(\mathbf{x})=(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}=\mathbf{D}^\top f(\mathbf{x})$ (根据公式可知**结果为一列向量**)

从上述计算过程可以看出，根据$\frac{\partial x_{kl}}{\partial x_{ij}}$可以计算出大部分的矩阵函数的Jacobian矩阵和梯度矩阵，但是对于复杂的矩阵函数，偏导$\frac{\partial x_{kl}}{\partial x_{ij}}$的计算就会比较困难。因此，产生了一种相对简单计算的方法：使用矩阵微分计算(**标量、向量和矩阵**)函数关于(**向量或矩阵**)变元的偏导。


##### 一阶实矩阵微分

矩阵微分记为$\mathrm{d}\mathbf{X}$，其定义为，

$$
\mathrm{d}\mathbf{X}=[\mathrm{d}X_{ij}]_{i=1,j=1}^{m,n}
$$

###### 性质

1. **转置不变** $\mathrm{d}(\mathbf{X}^\top)=(\mathrm{d}\mathbf{X})^\top$
2. **线性** $\mathrm{d}(\alpha\mathbf{X}\pm\beta\mathbf{Y})=\alpha\mathrm{d}\mathbf{X}\pm\beta\mathrm{d}\mathbf{Y}$
3. $\mathrm{d}(\mathbf{A})=\mathbf{0}$，常数矩阵$\mathbf{A}$。
4. $\mathrm{d}(\mathrm{tr}\mathbf{X})=\mathrm{tr}(\mathrm{d}\mathbf{X})$
5. $\mathrm{d}(\mathbf{AXB})=\mathbf{A}\mathrm{d}(\mathbf{X})\mathbf{B}$
6. 若有矩阵函数$\mathbf{U}=\mathbf{F}(\mathbf{X}),\mathbf{V}=\mathbf{G}(\mathbf{X}),\mathbf{W}=\mathbf{H}(\mathbf{X})$，则以下等式成立，

$$
d(\mathbf{UV})=(\mathrm{d}\mathbf{U})\mathbf{V}+\mathbf{U}(\mathrm{d}\mathbf{V})
$$

和

$$
d(\mathbf{UVW})=(\mathrm{d}\mathbf{U})\mathbf{VW}+\mathbf{U}(\mathrm{d}\mathbf{V})\mathbf{W}+\mathbf{U}\mathbf{V}(\mathrm{d}\mathbf{W})
$$

7. **迹不变性** 

$$\mathrm{d}(\mathrm{tr}(\mathbf{X}))=\mathrm{tr}(\mathrm{d} \mathbf{X})$$

8. **行列式微分** 

$$\mathrm{d}|\mathbf{X}|=|\mathbf{X}|\mathrm{tr}(\mathbf{X}^{-1}\mathrm{d}\mathbf{X})
$$

和

$$
\mathrm{d}\log|\mathbf{X}|=\mathrm{tr}(\mathbf{X}^{-1}\mathrm{d}\mathbf{X})
$$

9. **Kronecker积**

$$
\mathrm{d}(\mathbf{U}\otimes\mathbf{V})=\mathrm{d}(\mathbf{U})\otimes\mathbf{V}+\mathbf{U}\otimes\mathrm{d}(\mathbf{V})
$$

10. **Hadamard**

$$
\mathrm{d}(\mathbf{U}*\mathbf{V})=\mathrm{d}(\mathbf{U})*\mathbf{V}+\mathbf{U}*\mathrm{d}(\mathbf{V})
$$

11. **向量化**

$$\mathrm{d}(\mathrm{vec}(\mathbf{X}))=\mathrm{vec}(\mathrm{d}(\mathbf{X}))$$

12. **对数函数**

$$\mathrm{d}\log\mathbf{X}=\mathbf{X}^{-1} \mathrm{d}\mathbf{X}$$

13. **逆**

$$\mathrm{d}\mathbf{X}^{-1}=-\mathbf{X}^{-1} (\mathrm{d}\mathbf{X})\mathbf{X}^{-1}$$


###### 标量函数$f(\mathbf{x})$的向量变元$\mathbf{x}$全微分求偏导方法

考虑标量函数$f(\mathbf{x}),\mathbf{x}\in\mathbb{R}^n$的全微分，

$$
\mathrm{d}f(\mathbf{x})=\frac{\partial f(\mathbf{x})}{\partial x_1}\mathrm{d}x_1+\cdots+\frac{\partial f(\mathbf{x})}{\partial x_n}\mathrm{d}x_n
$$

写成内积形式，即

$$
\mathrm{d}f(\mathbf{x})=\left[\frac{\partial f(\mathbf{x})}{\partial x_1},\cdots,\frac{\partial f(\mathbf{x})}{\partial x_n}\right]\left[\begin{array}{cc}\mathrm{d}x_1\\\vdots\\\mathrm{d}x_n\end{array}\right]=\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}\mathrm{d}\mathbf{x}=\mathrm{tr}\left(\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}\mathrm{d}\mathbf{x}\right)
$$

显然，如果函数$f(\mathbf{x})$的微分可以写成

$$\mathrm{d}f(\mathbf{x})=\mathrm{tr}\left(\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}\mathrm{d}\mathbf{x}\right)$$

的形式，则$\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}$就是函数$f(\mathbf{x})$关于变元$\mathbf{x}$的Jacobian矩阵$\mathbf{D}_\mathbf{x}f(\mathbf{x})$，其转置$\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}=\left[\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}^\top}\right]^\top$即为梯度$\nabla_\mathbf{x}f(\mathbf{x})$。

###### 标量函数$f(\mathbf{X})$的矩阵量变元$\mathbf{X}$全微分求偏导方法

和向量一样，把变元矩阵的每个元素看作是一个变量，则标量函数$f(\mathbf{X}),\mathbf{X}\in\mathbb{R}^{m\times n}$的全微分可以写成如下形式，

$$
\begin{split}
\mathrm{d}f(\mathbf{X})&=\frac{\partial f(\mathbf{X})}{\partial \mathbf{x}_1}\mathrm{d}\mathbf{x}_1 +\cdots +\frac{\partial f(\mathbf{X})}{\partial \mathbf{x}_n}\mathrm{d}\mathbf{x}_n\\
&=\left[\frac{\partial f(\mathbf{x})}{\partial x_{11}},\cdots,\frac{\partial f(\mathbf{x})}{\partial x_{m1}}\right]\left[\begin{array}{cc}\mathrm{d}x_{11}\\\vdots\\\mathrm{d}x_{m1}\end{array}\right]+\cdots  + \left[\frac{\partial f(\mathbf{x})}{\partial x_{1n}},\cdots,\frac{\partial f(\mathbf{x})}{\partial x_{mn}}\right]\left[\begin{array}{cc}\mathrm{d}x_{1n}\\\vdots\\\mathrm{d}x_{mn}\end{array}\right]\\
&=\frac{\partial f(\mathbf{X})}{\partial \mathrm{vec}^\top(\mathbf{X})}\mathrm{d}(\mathrm{vec}(\mathbf{X}))\\
&=\mathbf{D}_{\mathrm{vec}\mathbf{X}} f(\mathbf{X})\mathrm{d}(\mathrm{vec}(\mathbf{X}))\\
&=[\mathrm{vec}(\mathbf{D}^\top_\mathbf{X}f(\mathbf{X}))]^\top  \mathrm{d}(\mathrm{vec}(\mathbf{X}))\\
&=\mathrm{tr}(\mathbf{D}_\mathbf{X}f(\mathbf{X}))\mathrm{d}\mathbf{X})
\end{split}
$$

易知：$\nabla_\mathbf{X}f(\mathbf{X})=\mathbf{D}^\top_\mathbf{X}f(\mathbf{X})$，即可求出梯度矩阵。

###### 求导方法

综上所述，若标量函数$f(\mathbf{x}),f(\mathbf{X})$在$\mathbf{x},\mathbf{X}$可微，则Jacobin矩阵存在，且可以通过以下形式计算，

$$
\begin{split}
\mathrm{d}f(\mathbf{x})&=\mathrm{tr}(\mathbf{D}_\mathbf{x}f(\mathbf{x}) \mathrm{d}\mathbf{x})=\mathrm{tr}(\nabla_\mathbf{x}f(\mathbf{x})^\top \mathrm{d}\mathbf{x})\\
\mathrm{d}f(\mathbf{X})&=\mathrm{tr}(\mathbf{D}_\mathbf{X}f(\mathbf{X}) \mathrm{d}\mathbf{X})=\mathrm{tr}(\nabla_\mathbf{X}f(\mathbf{X})^\top \mathrm{d}\mathbf{X})\\
\end{split}
$$

因此有以下通过对函数微分求偏导数的方法：将函数$f(\mathbf{X}),f(\mathbf{x})$求关于变元$\mathbf{X},\mathbf{x}$的微分(利用第1小节的性质)，并将结果整理成规范形式$\mathrm{d}f(\mathbf{x})=\mathrm{tr}(\mathbf{D}_\mathbf{x}f(\mathbf{x}) \mathrm{d}\mathbf{x})$或$\mathrm{d}f(\mathbf{X})=\mathrm{tr}(\mathbf{D}_\mathbf{X}f(\mathbf{X}) \mathrm{d}\mathbf{X})$。Jacobian矩阵的转置即为梯度矩阵。

因为最终要整理成规范形式，因此，需要用到迹的一些性质：

1. $a=\mathrm{tr}(a)$
2. $\mathrm{tr}(\mathbf{A}^\top)=\mathrm{tr}(\mathbf{A})$
3. $\mathrm{tr}(\alpha\mathbf{A}\pm\beta\mathbf{B})=\alpha\mathrm{tr}(\mathbf{A})\pm\beta\mathrm{tr}(\mathbf{B})$
4. $\mathrm{tr}(\mathbf{ABC})=\mathrm{tr}(\mathbf{BCA})=\mathrm{tr}(\mathbf{CAB})$
5. $\mathrm{tr}(\mathbf{AB})=\sum_{ij}A_{ji}B_{ij}$
6. $\mathrm{tr}(\mathbf{A^\top B})=\sum_{ij}A_{ij}B_{ij}=\mathrm{vec}(\mathbf{A})^\top\mathrm{vec}(\mathbf{B})$


###### 案例

1. 求实值函数$f(\mathbf{x})=\mathbf{x}^\top\mathbf{A}\mathbf{x}$的梯度向量与Jacobian矩阵。

使用微分法，凑规范形式如下，

$$
\begin{split}
\mathrm{d}(\mathbf{x}^\top\mathbf{A}\mathbf{x})&=\mathrm{tr}\left[(\mathrm{d}\mathbf{x}^\top)\mathbf{Ax} + \mathbf{x}^\top\mathbf{A}\mathrm{d}\mathbf{x}\right]\\
&=\mathrm{tr}[\mathbf{x}^\top\mathbf{A}^\top\mathrm{d}\mathbf{x}]+\mathrm{tr}[\mathbf{x}^\top\mathbf{A}\mathrm{d}\mathbf{x}]\\
&=\mathrm{tr}[\mathbf{x}^\top(\mathbf{A}+\mathbf{A}^\top)\mathrm{d}\mathbf{x}]
\end{split}
$$

因此可知，

$$
\begin{split}
\mathbf{D}_\mathbf{x}f(\mathbf{x})&=\mathbf{x}^\top(\mathbf{A}+\mathbf{A}^\top)\\
\nabla_\mathbf{x}f(\mathbf{x})&=(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}
\end{split}
$$

2. 求$\mathrm{tr}(\mathbf{X}^\top\mathbf{X})$的Jacobian矩阵和梯度矩阵

$$
\begin{split}
\mathrm{d}[\mathrm{tr}(\mathbf{X}^\top\mathbf{X})]&=\mathrm{tr}[(\mathrm{d}\mathbf{X}^\top)\mathbf{X}+\mathbf{X}^\top\mathrm{d}\mathbf{X}]\\
&=\mathrm{tr}[2\mathbf{X}^\top\mathrm{d}\mathbf{X}]
\end{split}
$$

因此可知，

$$
\begin{split}
\mathbf{D}_\mathbf{X}f(\mathbf{X})&=2\mathbf{X}^\top\\
\nabla_\mathbf{X}f(\mathbf{X})&=2\mathbf{X}
\end{split}
$$

3. 求$\mathrm{tr}(\mathbf{X}^\top\mathbf{AX})$的Jacobian矩阵和梯度矩阵

$$
\begin{split}
\mathrm{d}[\mathrm{tr}(\mathbf{X}^\top\mathbf{AX})]&=\mathrm{tr}[(\mathrm{d}\mathbf{X}^\top)\mathbf{AX}+\mathbf{X}^\top\mathbf{A}\mathrm{d}\mathbf{X}]\\
&=\mathrm{tr}[\mathbf{X}^\top(\mathbf{A}+\mathbf{A}^\top)\mathrm{d}\mathbf{X}]
\end{split}
$$

因此可知，

$$
\begin{split}
\mathbf{D}_\mathbf{X}f(\mathbf{X})&=\mathbf{X}^\top(\mathbf{A}+\mathbf{A}^\top)\\
\nabla_\mathbf{X}f(\mathbf{X})&=(\mathbf{A}+\mathbf{A}^\top)\mathbf{X}=\frac{\partial \mathrm{tr}(\mathbf{X}^\top\mathbf{AX})}{\partial\mathbf{X}}
\end{split}
$$

4. 求$\mathrm{tr}(\mathbf{A}\mathbf{X}^{-1})$的Jacobian矩阵和梯度矩阵

$$
\begin{split}
\mathrm{d}[\mathrm{tr}(\mathbf{A}\mathbf{X}^{-1})]&=\mathrm{tr}[(\mathbf{A}\mathrm{d}\mathbf{X}^{-1}]\\
&=-\mathrm{tr}[\mathbf{A}\mathbf{X}^{-1}(\mathrm{d}\mathbf{X})\mathbf{X}^{-1}]\\
&=\mathrm{tr}[-\mathbf{X}^{-1}\mathbf{A}\mathbf{X}^{-1}(\mathrm{d}\mathbf{X})]
\end{split}
$$

因此可知，

$$
\begin{split}
\mathbf{D}_\mathbf{X}f(\mathbf{X})&=-\mathbf{X}^{-1}\mathbf{A}\mathbf{X}^{-1}\\
\nabla_\mathbf{X}f(\mathbf{X})&=[-\mathbf{X}^{-1}\mathbf{A}\mathbf{X}^{-1}]^\top=\frac{\partial \mathrm{tr}(\mathbf{A}\mathbf{X}^{-1})}{\partial\mathbf{X}}
\end{split}
$$