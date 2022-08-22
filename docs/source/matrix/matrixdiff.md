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

1. **向量偏导算子**。
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

**梯度算子**指的是以列向量形式$\mathbf{x}=[x_1,...,x_m]^\top$定义的偏导算子，记为$\nabla_{\mathbf{x}}$，定义为

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