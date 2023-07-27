# 再生核希尔伯特空间



## 基本概念


### 泛函

&emsp;&emsp;首先来了解什么是泛函。**泛函**(Functional)通常是指定义域为函数集，而值域为实数或复数的映射。换句话说，泛函是输入为函数，输出为标量的映射。

>**定义0 (泛函)**. 假设有映射$F[y]: \mathcal{C}\rightarrow \mathbb{B}$，其中$\forall y(x)\in \mathcal{C}$，$\mathcal{C}$为函数集, $y$为$x$的函数，$x\in\mathcal{X}$,数集$\mathbb{B}\subseteq \mathbb{R}$，则称$F$是$y(x)$的泛函。一般记为积分形式
> $$
> J[y(x)]=\int_{x_0}^{x_1}F(x,y,y')dx
> $$

&emsp;&emsp;可以看出泛函是一种定义域为函数集，而值域为实数或复数的映射。

- **例1**. 两点间直线距离最短问题.

![两点间直线距离最短](../img/functional1.png)

&emsp;&emsp;如上图所示，从坐标原点$(0,0)$到点$(a,b)$的连接曲线为$y=y(x)$，曲线的弧长微分为$ds=\sqrt{1+\left( \frac{dy}{dx}\right)^2}dx$，总弧长为$s=\int_0^a (1+y'^2)^{1/2}dx$。可以看出$s$为一个标量，而等式右边为$y'(x)$的函数，即泛函，可记为$s(y')$。最终问题转化为：寻找曲线$y=y(x)$，使得泛函$s(y')$最小。

- **例2**. 最速降线问题

![最速降线问题](../img/functional2.png)

&emsp;&emsp;如上图所示，有一物体从原点$(0,0)$沿着曲线到达$A$点$(a,b)$，若忽略摩擦力，那么沿着什么形状的曲线路径下降时间最短？物体从点$(0,0)$运动到$P$点时，失去的势能为$mgy$，获得的动能为$mv^2/2$，由能量守恒定律可知，

$$
v^2=2gy
$$

在$P$点处，物体的运动速度为，

$$
v=\frac{ds}{dt}=\frac{\sqrt{1+y'^2}dx}{dt}
$$

其中$s$表示曲线弧长，$t$为运动时间。由上式可知，$dt=\frac{\sqrt{1+y'^2}dx}{v}$，将$v$代入，可得，

$$
dt=\sqrt{\frac{1+y'^2}{2gy}}dx
$$

最终可得物体从$O$点运动到$A$点所需的时间为(等式两边同时积分)，

$$
t=\int_0^a\sqrt{\frac{1+y'^2}{2gy}}dx=J[y]
$$

物体由$O$点运动到$A$点所需的时间$t$是$y(x)$的函数。问题最终转变为满足条件$y(0)=0,y(a)=b$的所有连续函数$y=y(x)$中，找出一个函数$y(x)$使得$J[y]$最小。

---

#### 泛函极值求解--变分法

&emsp;&emsp;对于引入的泛函问题$J[y]$，如何求其函数变时的极值$\hat{y}(x)$呢？一个可行的办法是使用变分法(Calculus of Variations)。传统的微积分中一个基本问题是寻找变最$x$的值使得函数$y(x)$的值大化(或最小化)。类似的，变分法能够寻找到一个函数$y(x)$使得泛函$F[y]$的值最大（或最小）。

&emsp;&emsp;传统的函数中，可以对变量$x$的微小变化$\epsilon$展开函数，即

$$
y(x+\epsilon)=y(x)+\frac{dy}{dx}\epsilon+O(\epsilon^2)
$$

类似的，对函数$y(x)$的微小变化$\epsilon \eta(x)$，可以展开泛函，即

$$
F[y(x)+\epsilon \eta(x)]=F[y(x)]+\epsilon\int\frac{\delta F}{\delta y(x)}\eta(x)dx+O(\epsilon^2)
$$

其中，$\frac{\delta F}{\delta y(x)}$为泛函$F[y]$关于$y(x)$的导数。上式可以看成函数展开的一个自然扩展。可以看出，泛函$F[y]$依赖$y$在所有点$x$的值。

&emsp;&emsp;现考虑更一般的泛函$F[y]$，

$$
F[y]=\int G(y(x),y'(x),x)dx
$$

其中，$y(x)$的值在积分边界上假定是固定的。现在对函数$y$的微小变化$\epsilon \eta(x)$，可以得到

$$
F[y(x)+\epsilon\eta(x)]=F[y(x)]+\epsilon\int\left\{\frac{\partial G}{\partial y}\eta(x)+\frac{\partial G}{\partial y'}\eta'(x)\right\}dx+O(\epsilon^2)
$$

注意：$\delta y=\epsilon \eta(x), \delta y'=\epsilon \eta'(x)$。对等式右边第二项使用分部积分可得，

$$
\begin{split}
\epsilon\int\frac{\partial G}{\partial y'}\eta'(x)dx&=\epsilon\int\frac{\partial G}{\partial y'}d\eta(x)\\
&=\left.\frac{\partial G}{\partial y'}\delta y\right|_{\textrm{boundary}_{\textrm{inf}}}^{\textrm{boundary}_{\textrm{sup}}}-\epsilon\int\eta(x)\frac{d}{dx}\left(\frac{\partial G}{\partial y'}\right)\\
&=-\epsilon\int\eta(x)\frac{d}{dx}\left(\frac{\partial G}{\partial y'}\right)\qquad (\textrm{在积分区域的边界}\delta y=0 )
\end{split}
$$

将上式结果代回，可得

$$
F[y(x)+\epsilon\eta(x)]=F[y(x)]+\epsilon\int\left\{\frac{\partial G}{\partial y}-\frac{d}{dx}\left(\frac{\partial G}{\partial y'}\right)\right\}\eta(x)dx+O(\epsilon^2)
$$

若泛函在$y(x)$取得极值，则必有，

$$
\boxed{\frac{\partial G}{\partial y}-\frac{d}{dx}\left(\frac{\partial G}{\partial y'}\right)=0}
$$

这便是变分法的核心公式，也称之为Euler-Lagrange方程。有了该公式，就可以找出所寻求的极值点$\hat{y}(x)$。一般来说，Euler方程是一个二阶微分方程，$y(x)$的通解中含有的两个待定常数刚好通过两个边界条件确定。



#### 含参变量的定积分

&emsp;&emsp;变分法中，对于泛函求导问题（求导与积分先后顺序问题），可以参考含变量的定积分。含变量的定积分一般具有如下形式，

$$
I(y)=\int_a^b f(x,y)dx
$$

>**定理1**. 若函数$f(x,y)$在闭区域$a\le x\le b,c\le y\le d$上连续，则$I[y]$在闭区域$c\le y\le d$连续。

>**定理2**.  若函数$f(x,y),f_y'(x,y)$都在闭区域$a\le x\le b,c\le y\le d$上连续，则$I[y]$在闭区域$c\le y\le d$上具有连续导数$I'[y]=\int_a^b f'_y(x,y)dx$.

证明过程可参考《数学分析》ISBN：9787040427806。

---



#### 例子

- **例1**. 求解两点间最短路径。

&emsp;&emsp;该问题的泛函已知为，

$$
s=\int_0^a (1+y'^2)^{1/2}dx
$$

使用变分法，令$G(y,y',x)=\sqrt{1+y'^2}$，使用Euler-Lagrange方程来寻找$s$有极值的函数$y(x)$。注意到，

$$
\frac{\partial G}{\partial y}=0,\quad\frac{\partial G}{\partial y'}=\frac{y'}{\sqrt{1+y'^2}}
$$

代入Euler-Lagrange方程，可知，

$$
\frac{d}{dx}\left(\frac{y'}{\sqrt{1+y'^2}} \right)=0
$$

解此微分方程，可知$y(x)$满足直线方程，即，

$$
\boxed{y=kx+c}
$$

通过边界点，可以计算出$k,c$。由此，通过变分法，我们得到了结论：两点之间直线距离最短。

- **例2**. 最速降线问题。

&emsp;&emsp;该问题的泛函上节已知为

$$
t=\int_0^a\sqrt{\frac{1+y'^2}{2gy}}dx
$$

利用变分法，令$G(y,y',x)=\sqrt{\frac{1+y'^2}{2gy}}$，可求得Euler-Lagrange方程的两个偏导数，

$$
\frac{\partial G}{\partial y}=-\frac12\sqrt{\frac{1+y'^2}{y^3}},\quad\frac{\partial G}{\partial y'}=\frac{y'}{\sqrt{y(1+y'^2)}}
$$

可得Euler方程如下，

$$
\frac12\sqrt{\frac{1+y'^2}{y^3}}+\frac{d}{dx}\left(\frac{y'}{\sqrt{y(1+y'^2)}}\right)=0
$$

注意到，

$$
\frac{d}{dx}\left[G-y'\frac{\partial G}{\partial y'}\right]=y'\frac{\partial G}{\partial y}+y''\frac{\partial G}{\partial y'}-y''\frac{\partial G}{\partial y'}-y'\frac{d}{dx}\left(\frac{\partial G}{\partial y'}\right)=0
$$

因此有，

$$
G-y'\frac{\partial G}{\partial y'}=C
$$

做三角代换可得，

$$
y=2r\sin^2\frac{\theta}{2}=r(1-\cos\theta)
$$

上式对$\theta$求导，可得$x=r(\theta-\sin\theta)+x_0$。根据曲线过原点$(0,0)$及$(p,q)$可解出$x_0=0$以及$r$，最终结果为，

$$
\boxed{\left\{\begin{array}{l}x=r(\theta-\sin\theta)\\ y=r(1-\cos\theta)\end{array} \right.}
$$


解此方程，最终得到

[参考1](https://zhuanlan.zhihu.com/p/139018146)


## 线性算子

>&emsp;&emsp;**定义1 (线性算子)**. 一个函数$A:\mathcal{F}\rightarrow \mathcal{G}$称之为线性算子，当且仅当以下条件成立：
>> - 齐次性(homogeneity): $A(\alpha f)=\alpha (Af),\quad \forall \alpha\in \mathbb{K},f\in \mathcal{F}$
>> - 可加性(additivity): $A(f+g)=Af+Ag,\quad\forall f,g\in\mathcal{F}$
>
>其中$\mathcal{F,G}$为定义在$\mathbb{K}$上的赋范向量空间（例如：$\mathcal{X}\subset \mathbb{R}\rightarrow \mathbb{R} $ 映射函数的Banach空间并定义$L_p$范数）。

&emsp;&emsp;**例1** 令$\mathcal{F}$为一个内积空间，对于$g\in\mathcal{F}$，算子$A_g :\mathcal{F}\rightarrow\mathbb{K}$， $A_g(f):=\langle f,g\rangle_\mathcal{F}$是一个线性算子。注意到算子$A_g$的像是一个潜在的域$\mathbb{K}$，而这正是定义在$\mathbb{K}$上的一个平凡的赋范线性空间(normed linear space over itself)。这样的标题值也称之为$\mathcal{F}$空间的泛函(functionals)。

>&emsp;&emsp;**定义2 (连续性)**. 一个函数$A:\mathcal{F}\rightarrow \mathcal{G}$在$f_0\subset \mathcal{F}$是连续的，如果对于任意$\epsilon >0$，总存在一个$\delta =\delta(\epsilon,f_0)>0$，满足

$$
\Vert f-f_0\Vert_\mathcal{F}<\delta,\quad \textrm{implies}\quad\Vert Af-Af_0\Vert_\mathcal{G}<\epsilon.
$$

&emsp;&emsp;当$A$在$\mathcal{F}$的每一个点都是连续的，则$A$在$\mathcal{F}$是**连续的**。换句话说，$\mathcal{F}$中的一个收敛序列映射到$\mathcal{G}$中的一个收敛序列。



## 再生核Hilbert空间

&emsp;&emsp;一个特殊的泛函：求值泛函(Dirac evaluation functional).

> **定义1 (Evaluation Functional)**. Let $\mathcal{H}$ be a Hilbert space of functions $f:\mathcal{X}\rightarrow \mathbb{R}$, defined on a non-empty set $\mathcal{X}$. For a fixed $x\in\mathcal{X}$，map $\delta_x :\mathcal{H}\rightarrow\mathbb{R}$，$\delta_x :f\rightarrow  f(x)$ is called the (Dirac) evaluation functional at $x$.

显然，求值泛函$\delta_x$是一个线性泛函，因为对于$\forall f,g\in\mathcal{H}, \forall \alpha,\beta\in\mathbb{R}$以下等式成立，

$$
\delta_x(\alpha f+\beta g)=(\alpha f+\beta g)(x)=\alpha f(x)+\beta g(x)=\alpha \delta_x(f)+\beta\delta_x(g)
$$

> **定义2 (RKHS)**. 设$\mathcal{H}$是一个定义在非空集合$\mathcal{X}$上的函数$f:\mathcal{X}\rightarrow \mathbb{R}$构成的Hilbert空间，若函数$\kappa:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$满足：
>1. $\forall x \in \mathcal{X},\kappa(\cdot,x)\in\mathbb{R}$;
>2. $\forall x \in \mathcal{X},\forall f \in \mathcal{H},\langle f,\kappa(\cdot,x)\rangle_\mathcal{H}=f(x)$;
>3. 特别地，$\forall x,y\in\mathcal{X}$，有$\kappa(x,y)=\langle \kappa(\cdot,x),\kappa(\cdot,y)\rangle_\mathcal{H}$;
>
>其中，$\langle\cdot,\cdot\rangle_\mathcal{H}$是内积。则称$k$为$\mathcal{H}$的再生核函数，$\mathcal{H}$为再生核Hilbert空间。


[参考1](http://songcy.net/posts/story-of-basis-and-kernel-part-2/)[参考2](https://www.cnblogs.com/zhangcn/p/13289236.html)