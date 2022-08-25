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
\mathrm{d}\mathbf{X}=[dX_{ij}]_{i=1,j=1}^{m,n}
$$

###### 性质

1. **转置不变** $\mathrm{d}(\mathbf{X}^\top)=(\mathrm{d}\mathbf{X})^\top$
2. **线性** $\mathrm{d}(\alpha\mathbf{X}\pm\beta\mathbf{Y})=\alpha\mathrm{d}\mathbf{X}\pm\beta\mathrm{d}\mathbf{Y}$
3. $\mathrm{d}(\mathbf{A})=\mathbf{0}$，常数矩阵$\mathbf{A}$。