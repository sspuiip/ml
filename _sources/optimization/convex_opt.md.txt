### 凸优化

#### 凸函数

所谓的凸函数是指：定义在凸集$\mathcal{S}$的任意变量$\mathbf{x},\mathbf{y}$，函数$f:\mathbb{R}^n\rightarrow \mathbb{R}$对任意$0\le\theta\le 1$都有以下不等式成立，

$$
f(\theta \mathbf{x}+(1-\theta)\mathbf{y})\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})
$$

，则该函数$f$是凸函数。

凸集指的是：集合$\mathcal{S}$中任意两点$\mathbf{x},\mathbf{y}\in\mathcal{S}$，连接它们的线段也在集合集合$\mathcal{S}$内，则该集合为凸集，即

$$
\mathbf{x,y}\in \mathcal{S},\quad \theta\in[0,1]\quad\Rightarrow \theta\mathbf{x}+(1-\theta)\mathbf{y}\in\mathcal{S}
$$

##### 判定方法

- **一阶充要条件**

    1. 若定义在凸集$\mathcal{S}$上的函数$f(\mathbf{x})$一阶可微且满足以下条件，

    $$
    f(\mathbf{y})\ge f(\mathbf{x})+\langle\nabla_\mathbf{x} f(\mathbf{x}),\mathbf{y}-\mathbf{x}\rangle
    $$

    则$f(\mathbf{x})$为凸函数。

    2. $f(\mathbf{x})$凸$\quad\Leftrightarrow\quad \langle\nabla_\mathbf{x}f(\mathbf{x})-\nabla_\mathbf{x}f(\mathbf{y}),\mathbf{x}-\mathbf{y}\rangle\ge 0$

    3. $f(\mathbf{x})$严凸$\quad\Leftrightarrow\quad \langle\nabla_\mathbf{x}f(\mathbf{x})-\nabla_\mathbf{x}f(\mathbf{y}),\mathbf{x}-\mathbf{y}\rangle> 0$


    4. $f(\mathbf{x})$强凸$\quad\Leftrightarrow\quad \langle\nabla_\mathbf{x}f(\mathbf{x})-\nabla_\mathbf{x}f(\mathbf{y}),\mathbf{x}-\mathbf{y}\rangle\ge \mu\lVert \mathbf{x}-\mathbf{y} \rVert_2^2$

- **二阶充要条件**

    1. $f(\mathbf{x})$凸$\quad\Leftrightarrow\quad \mathbf{H}[f(\mathbf{x})]\succeq 0$

    2. $f(\mathbf{x})$严凸$\quad\Leftrightarrow\quad \mathbf{H}[f(\mathbf{x})]\succ 0$

##### 保凸运算与性质判定

1. $f(\mathbf{x})$是凸函数，当且仅当它在所有线段上是凸的。

2. 非负线性运算仍为凸函数：若$a,b\ge 0$，$f_1(\mathbf{x}), f_2(\mathbf{x})$是凸函数，则$af_1(\mathbf{x})+bf_2(\mathbf{x})$仍为凸函数。

3. 凸函数的无穷求和、积分仍是凸函数。

4. 凸函数各点的上确界为凸函数：$f_a(\mathbf{x})$凸$\quad\Rightarrow \sup f_a(\mathbf{x})$仍为凸函数。

5. 凸函数的仿射变换仍为凸函数：$f(\mathbf{x})$凸$\quad\Rightarrow  f(\mathbf{Ax}+\mathbf{b})$仍为凸函数。

6. 向量的所有范数除$L_0$外都是凸函数。

#### 凸优化问题

凸优化问题是形如，

$$
\begin{split}
\min\limits_{\mathbf{x}} \quad &f_0(\mathbf{x})\\
\mathrm{s.t.}\quad &f_i(\mathbf{x})\leq0,i=1,...,m\\
\qquad &h_i(\mathbf{x})=0,i=1,...,p
\end{split}
$$

的问题，其中$f_0,...,f_m$是凸函数。对比优化问题，凸优化有3个附加的要求：

1. 目标函数必须是凸的
2. 不等式约束函数必须是凸的
3. 等式约束必须是仿射的

