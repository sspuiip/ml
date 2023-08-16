# 凸优化问题

## 凸函数

&emsp;&emsp;所谓的凸函数是指：定义在凸集$\mathcal{S}$的任意变量$\pmb{x},\pmb{y}$，函数$f:\mathbb{R}^n\rightarrow \mathbb{R}$对任意$0\le\theta\le 1$都有以下不等式成立，

$$
f(\theta \pmb{x}+(1-\theta)\pmb{y})\le\theta f(\pmb{x})+(1-\theta)f(\pmb{y})
\tag{1}
$$

，则该函数$f$是凸函数。

&emsp;&emsp;凸集指的是：集合$\mathcal{S}$中任意两点$\pmb{x},\pmb{y}\in\mathcal{S}$，连接它们的线段也在集合集合$\mathcal{S}$内，则该集合为凸集，即

$$
\pmb{x,y}\in \mathcal{S},\quad \theta\in[0,1]\quad\Rightarrow \theta\pmb{x}+(1-\theta)\pmb{y}\in\mathcal{S}
\tag{2}
$$

### 判定方法

- **一阶充要条件**

    1. 若定义在凸集$\mathcal{S}$上的函数$f(\pmb{x})$一阶可微且满足以下条件，

    $$
    f(\pmb{y})\ge f(\pmb{x})+\langle\nabla_\pmb{x} f(\pmb{x}),\pmb{y}-\pmb{x}\rangle
    \tag{3}
    $$

    则$f(\pmb{x})$为凸函数。

    2. $f(\pmb{x})$凸$\quad\Leftrightarrow\quad \langle\nabla_\pmb{x}f(\pmb{x})-\nabla_\pmb{x}f(\pmb{y}),\pmb{x}-\pmb{y}\rangle\ge 0$

    3. $f(\pmb{x})$严凸$\quad\Leftrightarrow\quad \langle\nabla_\pmb{x}f(\pmb{x})-\nabla_\pmb{x}f(\pmb{y}),\pmb{x}-\pmb{y}\rangle> 0$


    4. $f(\pmb{x})$强凸$\quad\Leftrightarrow\quad \langle\nabla_\pmb{x}f(\pmb{x})-\nabla_\pmb{x}f(\pmb{y}),\pmb{x}-\pmb{y}\rangle\ge \mu\lVert \pmb{x}-\pmb{y} \rVert_2^2$

- **二阶充要条件**

    1. $f(\pmb{x})$凸$\quad\Leftrightarrow\quad \pmb{H}[f(\pmb{x})]\succeq 0$

    2. $f(\pmb{x})$严凸$\quad\Leftrightarrow\quad \pmb{H}[f(\pmb{x})]\succ 0$

### 保凸运算与性质判定

&emsp;&emsp;1. $f(\pmb{x})$是凸函数，当且仅当它在所有线段上是凸的。

&emsp;&emsp;2. 非负线性运算仍为凸函数：若$a,b\ge 0$，$f_1(\pmb{x}), f_2(\pmb{x})$是凸函数，则$af_1(\pmb{x})+bf_2(\pmb{x})$仍为凸函数。

&emsp;&emsp;3. 凸函数的无穷求和、积分仍是凸函数。

&emsp;&emsp;4. 凸函数各点的上确界为凸函数：$f_a(\pmb{x})$凸$\quad\Rightarrow \sup f_a(\pmb{x})$仍为凸函数。

&emsp;&emsp;5. 凸函数的仿射变换仍为凸函数：$f(\pmb{x})$凸$\quad\Rightarrow  f(\pmb{Ax}+\pmb{b})$仍为凸函数。

&emsp;&emsp;6. 向量的所有范数除$L_0$外都是凸函数。

## 凸优化问题

&emsp;&emsp;**无约束凸函数优化问题**是指：优化目标为凸函数且定义域为凸集的优化问题。

&emsp;&emsp;**带约束凸优化问题**是形如，

$$
\begin{equation}
\begin{split}
\min\limits_{\pmb{x}} \quad &f_0(\pmb{x})\\
\mathrm{s.t.}\quad &f_i(\pmb{x})\leq0,i=1,...,m \\
\qquad &h_i(\pmb{x})=0,i=1,...,p
\end{split}
\tag{4}
\end{equation}
$$

的问题，其中$f_0,...,f_m$是凸函数。对比优化问题，凸优化有3个附加的要求：

&emsp;&emsp;1. 目标函数必须是凸的

&emsp;&emsp;2. 不等式约束函数必须是凸的

&emsp;&emsp;3. 等式约束必须是仿射的

&emsp;&emsp;**定理**：无约束凸函数$f(\pmb{x})$的任何局部极小点$\pmb{x}^*$都是该函数的一个全局最小点。若$f(\pmb{x})$可微，则$\frac{\partial f}{\partial\pmb{x}}=0$的平稳点$\pmb{x}^*$是$f(\pmb{x})$的一个全局最小点。

&emsp;&emsp;该定理表明：如果任何一个约束极小化问题可以*转换*为一个无约束凸优化问题，则无约束凸优化问题的任何一个局部极小点都是原约束极小化问题的一个全局极小点。

&emsp;&emsp;**引理**：若$f(\pmb{x})$强凸，则最小化问题$\min\limits_{\pmb{x}}f(\pmb{x})$是可解的，且惟一，并有，

$$\tag{5}
f(\pmb{x})\ge f(\pmb{x}^*)+\frac12\mu\lVert\pmb{x}-\pmb{x}^*\rVert_2^2,\quad\forall \pmb{x}\in\mathcal{S}
$$

其中，$\mu$是强凸函数$f(\pmb{x})$的凸性参数。

&emsp;&emsp;以上定理说明，约束优化问题的解决方向是：<font color="blue">将约束优化问题转换成无约束的凸优化问题</font>。

### Lagrangian乘子法


&emsp;&emsp;现**假设**约束极小化问题如下：

&emsp;&emsp;1. 不等式约束均为凸函数

&emsp;&emsp;2. 等式约束具有仿射函数形式$\pmb{h}(\pmb{x})=\pmb{Ax}-\pmb{b}$。

&emsp;&emsp;3. $f_0(\pmb{x})$可微，但不是凸函数。

&emsp;&emsp;使用**Lagrangian乘子法**，可以将约束优化问题转换为无约束优化问题，

$$
\min L(\pmb{x},\pmb{\lambda},\pmb{\nu})=f_0(\pmb{x})+\sum_{i=1}^m\lambda_if_i(\pmb{x})+\sum_{i=1}^p\nu_ih_i(\pmb{x})
\tag{6}
$$

&emsp;&emsp;其中：

&emsp;&emsp;- $L(\pmb{x},\pmb{\lambda},\pmb{\nu})$为Lagrangian函数

&emsp;&emsp;- $\lambda_i,\nu_i$为Lagrangian乘子，向量$\pmb{\lambda,\nu}$为Lagrangian乘子向量，并约束$\pmb{\lambda}\ge 0$

&emsp;&emsp;- $\pmb{x}$为优化变量,决策变量或原始变量

&emsp;&emsp;- $\pmb{\lambda}$为对偶变量

&emsp;&emsp;- 原约束优化问题称为原始问题

&emsp;&emsp;- 使用乘子法转换的无约束优化问题(6)称为对偶问题。


&emsp;&emsp;在式(6)中，当$\lambda_i$取较大值时，该式第2项可能趋于负无穷大，从而导致$L(\pmb{x},\pmb{\lambda},\pmb{\nu})$趋于负无穷大。因此，需要将Lagrangian函数极大化，

$$
J_1(\pmb{x})=\max\limits_{\pmb{\lambda}\succeq 0,\pmb{\nu}}L(\pmb{x,\lambda,\nu})=\max\limits_{\pmb{\lambda}\succeq 0,\pmb{\nu}}\left( f_0(\pmb{x})+\sum_{i=1}^m\lambda_if_i(\pmb{x})+\sum_{i=1}^p\nu_ih_i(\pmb{x})\right)\tag{7}
$$

&emsp;&emsp;但$J_1(\pmb{x})$仍存在一个问题：无法避免约束$f_i(\pmb{x})>0$，这将导致$J_1(\pmb{x})$正无穷大，即

$$
J_1(\pmb{x})=\left\{ \begin{array}{ll}f_0(\pmb{x}),& 若\pmb{x}满足原始全部约束\\(f_0(\pmb{x}),+\infty),& 其它 \end{array}\right.
$$

&emsp;&emsp;因此，为得到约束优化的极小解$\min\limits_{\pmb{x}}f_0(\pmb{x})=f_0(\pmb{x}^*)$，必须将$J_1(\pmb{x})$极小化，即

$$
J_P(\pmb{x})=\min\limits_{\pmb{x}}J_i(\pmb{x})=\min\limits_{\pmb{x}}\max\limits_{\pmb{\lambda}\succeq 0,\pmb{\nu}}L(\pmb{x,\lambda,\nu})
$$

&emsp;&emsp;这是一个极小-极大问题，解为$L(\pmb{x,\lambda,\nu})$的上确界(supremum)即最小上界。

&emsp;&emsp;综上可知，原始约束极小化问题的最优值为

$$
p^*=J_P(\pmb{x})=\min\limits_{\pmb{x}}f_0(\pmb{x})=f_0(\pmb{x}^*)
$$

### 对偶方法

&emsp;&emsp;极小-极大问题仍然存在问题：一个非凸目标函数(前一节的假设3)的极小化不能转换成另一个凸函数的极小化。

&emsp;&emsp;我们知道凸函数$f(\pmb{x})$的极小化与凹函数$-f(\pmb{x})$的极大化等价。基于这一对偶关系，可以将非凸目标函数的极小化转化为凹函数的极大化。这一方法也称为**Lagrangian对偶法**。

&emsp;&emsp;首先，将Lagrangian函数构造一个极小化目标函数，

$$
J_2(\pmb{\lambda,\nu})=\min\limits_{\pmb{x}}\left(f_0(\pmb{x})+\sum_{i=1}^m\lambda_if_i(\pmb{x})+\sum_{i=1}^p\nu_ih_i(\pmb{x}) \right)
$$

&emsp;&emsp;与上一节类似，该函数的极小值为，

$$
\min\limits_{\pmb{x}}L(\pmb{x,\lambda,\nu})=\left\{\begin{array}{ll}\min\limits_{\pmb{x}}f_0(\pmb{x}),&\pmb{x}\mathrm{满足所有约束}\\(-\infty,\min\limits_{\pmb{x}}f_0(\pmb{x}),&其它.\end{array}\right.
$$

&emsp;&emsp;其极大化函数为，

$$
J_D(\pmb{\lambda,\nu})=\max\limits_{\pmb{\lambda}\succeq0,\pmb{\nu}}J_2(\pmb{\lambda,\nu})=\max\limits_{\pmb{\lambda}\succeq0,\pmb{\nu}}\min\limits_{\pmb{x}}L(\pmb{x,\lambda,\nu})
$$

&emsp;&emsp;称为原问题的对偶目标函数，它是Lagrangian函数的极大-极小问题。其解为该函数的下确界（最大下界）。

&emsp;&emsp;<font color="red">注意</font>：对偶目标函数$J_D(\pmb{\lambda,\nu})$是下无界的（下界为$-\infty$），因此$J_D(\pmb{\lambda,\nu})$是变元$\pmb{x}$的凹函数（<font color="#feda0f">因为对偶函数是一族关于$(\lambda,\nu)$的仿射函数的逐点下确界</font>），即使$f_0(\pmb{x})$不是凸函数。

&emsp;&emsp;对偶目标函数的最优值为，

$$
d^*=J_D(\pmb{\lambda^*,\nu^*})\le\min\limits_{\pmb{x}}f_0(\pmb{x})=p^*
$$

&emsp;&emsp;最优原始值与最优对偶值之差$d^*-p^*$称为原问题与对偶问题之间的**对偶间隙**，最优对偶值是原始最优值的下界。

&emsp;&emsp;- $d^*\le p^*$：弱对偶性

&emsp;&emsp;- $d^*= p^*$：强对偶性


### KKT条件

&emsp;&emsp;假设$\pmb{x}^*,\pmb{\lambda^*,\nu^*}$分别为具有对偶间隙$\epsilon=0$的原始最优点和对偶最优点。

&emsp;&emsp;由于$\pmb{x}^*$使得Lagrangian目标函数$L(\pmb{x},\pmb{\lambda}^*,\pmb{\nu}^*)$在所有原始可行点$\pmb{x}$中最小化，所以Lagrangian目标函数$L(\pmb{x},\pmb{\lambda}^*,\pmb{\nu}^*)$在点点$\pmb{x}^*$的梯度向量必为0，

$$
\nabla f_0(\pmb{x}^*)+\sum_i\lambda_i^*\nabla f_i(\pmb{x}^*)+\sum_i\nu_i^*\nabla h_i(\pmb{x}^*)=\pmb{0}
$$

&emsp;&emsp;因此，Lagrangian对偶无约束优化问题的KKT条件（局部极小解的一阶必要条件）为，

$$
\begin{split}
f_i(\pmb{x}^*)&\le0, \quad i=1,\cdots,m\\
h_i(\pmb{x}^*)&=0, \quad i=1,\cdots,p\\
\lambda_i^*&\ge0, \quad i=1,\cdots,m\\
\lambda_i^*f_i(\pmb{x}^*)&=0, \quad i=1,\cdots,m\\
\nabla f_0(\pmb{x}^*)+\sum_i\lambda_i^*&\nabla f_i(\pmb{x}^*)+\sum_i\nu_i^*\nabla h_i(\pmb{x}^*)=\pmb{0}
\end{split}
$$

&emsp;&emsp;- 前2个条件为约束条件。

&emsp;&emsp;- 第3个条件是Lagrangian乘子的非负约束，是Lagrangian乘子法的关键约束。

&emsp;&emsp;- 第4个条件是Lagrangian对偶法的另一个关键约束。这意味着对于$f_i(\pmb{x})>0$的约束，Lagrangian乘子$\lambda_i$必须等于0。


### 强对偶性（Slater定理）

&emsp;&emsp;判别强对偶性是否成立的一种简单方法是Slater定理。

&emsp;&emsp;**定义**:原始不等式约束的可行域的相对内域为

$$
\mathrm{relint}(\mathcal{F})=\{\pmb{x}|f_i(\pmb{x})<0,i=1,\cdots,m;h_i(\pmb{x})=0,i=1,\cdots,p \}
$$

&emsp;&emsp;**Slater条件**：优化过程中，迭代点位于可行域的内域的约束规定称为Slater条件。

&emsp;&emsp;**Slater定理**：如果Slater条件满足，并且原始不等式优化问题是凸优化问题，则对偶无约束优化问题的最优值$d^*$与原始优化问题的最优值$p^*$相等，

$$
d^*=p^*
$$

即强对偶性成立。





