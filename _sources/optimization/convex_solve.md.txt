### 优化问题求解

#### 下降法

基本思想：利用优化序列

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+\mu_k\Delta\mathbf{x}_k,\quad k=1,2,\cdots\tag{1}
$$

寻找最优点$\mathbf{x}_{opt}$。$\mu_k$为第$k$次迭代的步长，$\Delta\mathbf{x}_k$为搜索方向其值为一个向量$\Delta\mathbf{x}\in\mathbb{R}^n$。最小化算法要求迭代过程中目标函数是下降的，

$$
f(\mathbf{x}_{k+1})<f(\mathbf{x}_k)
$$

所以该方法称为<font color='red'>**下降法**</font>。


##### 最速下降法

由Taylor公式可知，

$$
f(\mathbf{x}_{k+1})- f(\mathbf{x}_k)\approx\nabla f(\mathbf{x}_k)^\top\Delta\mathbf{x}_k\tag{2}
$$

显然，当$0\le\theta\le \pi/2$, 令

$$
\Delta\mathbf{x}_k=-\nabla f(\mathbf{x}_k)\cos\theta\tag{3}
$$

必然有，$f(\mathbf{x}_{k+1})<f(\mathbf{x}_k)$成立。

- 取$\theta=0$，则$\Delta\mathbf{x}_k=-\nabla f(\mathbf{x}_k)$，即搜索方向为负梯度方向，步长为$\lVert \nabla f(\mathbf{x}_k)\rVert_2^2$，故下降方向具有最大的下降步伐。与之对应的下降法称为<font color='red'>**最速下降法**</font>。

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+\mu_k\nabla f(\mathbf{x}_k)\tag{4}
$$

##### Newton法

Taylor公式展开至二阶有，

$$
f(\mathbf{x}_{k+1})\approx f(\mathbf{x}_k)+\nabla f(\mathbf{x}_k)^\top\Delta\mathbf{x}_k+\frac12(\Delta\mathbf{x}_k)^\top\nabla^2f(\mathbf{x}_k)(\Delta\mathbf{x}_k)\tag{5}
$$

显然最优化下降方向应该是让二阶展开式最得最小值的方向，即

$$
\min\limits_{\Delta\mathbf{x}_k}\left[f(\mathbf{x}_k)+\nabla f(\mathbf{x}_k)^\top\Delta\mathbf{x}_k+\frac12(\Delta\mathbf{x}_k)^\top\nabla^2f(\mathbf{x}_k)(\Delta\mathbf{x}_k)\right]\tag{6}
$$

对二阶展开式求导，可知，

$$
\frac{\partial f(\mathbf{x}_k+\Delta\mathbf{x}_k)}{\partial \Delta\mathbf{x}_k}=\nabla f(\mathbf{x}_k)+\nabla^2f(\mathbf{x}_k)\Delta\mathbf{x}_k=0\tag{7}
$$

则有最优搜索方向，

$$
\Delta\mathbf{x}_k=-\nabla^2f(\mathbf{x}_k)\nabla f(\mathbf{x}_k)\tag{8}
$$

该下降方向也称之为Newton步或Newton下降方向，记为$\Delta\mathbf{x}_{nt}$，相应的方法称为<font color='red'>**Newton法**</font>。

#### 梯度投影法

梯度下降法中变元是无约束的。若有约束$\mathbf{x}\in\mathcal{C}$，则梯度下降法中的更新公式应用投影代替，

$$
\mathbf{x}_{k+1}=\mathcal{P}_\mathcal{C}(\mathbf{x}_k-\mu_k\nabla f(\mathbf{x}_k))\tag{9}
$$

这一算法称为梯度投影法，也称为投影梯度法。投影算子$\mathcal{P}_\mathcal{C}(\mathbf{y})$定义为

$$
\mathcal{P}_\mathcal{C}(\mathbf{y})=\arg\min\limits_{\mathbf{x}\in\mathcal{C}}\frac12\lVert \mathbf{x}-\mathbf{y}\rVert_2^2\tag{10}
$$

**例**. 到超平面$\mathcal{C}=\{\mathbf{x}|\mathbf{a}^\top\mathbf{x}=b\}$的投影，

$$
\mathcal{P}_\mathcal{C}(\mathbf{x})=\mathbf{x}+\frac{b-\mathbf{a}^\top\mathbf{x}}{\lVert\mathbf{a}\rVert_2^2}\mathbf{a}
$$

求解过程：投影问题为如下优化问题，

$$
\mathcal{P}_\mathcal{C}(\mathbf{x})=\arg\min\limits_{\mathbf{z}\in\mathcal{C}}\frac12\lVert \mathbf{x}-\mathbf{z}\rVert_2^2 \quad s.t.\quad \mathbf{a}^\top\mathbf{x}-b=0
$$

则Lagrangian函数为，

$$
L(\mathbf{x},\lambda)=\frac12\lVert \mathbf{x}-\mathbf{z}\rVert_2^2  +\lambda( \mathbf{a}^\top\mathbf{z}-b)
$$

对Lagrangian函数求偏导并令其等于0，可得

$$
\begin{split}
\frac{\partial L}{\partial \mathbf{z}}&=\mathbf{z}-\mathbf{x}+\lambda \mathbf{a}=0\\
\frac{\partial L}{\partial \lambda}&=\mathbf{a}^\top\mathbf{z}-b=0\\
\end{split}
$$

解上述方程组，将$\mathbf{z}=\mathbf{x}-\lambda \mathbf{a}$代入$\mathbf{a}^\top\mathbf{z}-b=0$，可得，

$$
\lambda = \frac{\mathbf{a}^\top\mathbf{x}-b}{\mathbf{a}^\top\mathbf{a}}
$$

再将$\lambda$代入$\mathbf{z}-\mathbf{x}+\lambda \mathbf{a}=0$，可得

$$
\mathbf{z}=\mathbf{x}+\frac{b-\mathbf{a}^\top\mathbf{x}}{\mathbf{a}^\top\mathbf{a}}\mathbf{a}
$$


#### 共轭梯度下降法

最速下降法的存在一个问题就是收敛速度过慢，因为已迭代的$\mathbf{x}$会来回振荡，从而导致收敛太慢。

Newton法虽然收收敛较快，但仍需要计算Hessian矩阵的逆，因此计算代价太高。


为了加速最速下降法的收敛速度和避免Newton法的Hessian逆矩阵计算，提出了共轭梯度下降法。

![alt Conjugate Gradient Descent](../img/conj_desc.png)

与前面两种下降方法类似，共轭梯度下降也是通过迭代来寻找最优点，即

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+\alpha_k\mathbf{d}_k\tag{11}
$$

**不同之处**在于，每次迭代的下降方向向量$\mathbf{d}_i$与其它任何一次方向向量$\mathbf{d}_j,j\neq i$都是$\mathbf{A}$-共轭的；此外，$\alpha_i$是$\min\limits_{\alpha}f(\mathbf{x}_{i-1}+\alpha\mathbf{d}_i)$的最优值。

为了简要描述共轭的思想，以上图为例，坐标轴可以指定为搜索方向。第一步沿着水平方向到达$\mathbf{x}^*$的$x_1$分量部分。第二步没着垂直方向到达$\mathbf{x}^*$的$x_2$分量部分，然后结束搜索过程就可以确定$\mathbf{x}^*$的值。如果定义$\mathbf{e}_i=\mathbf{x}^*-\mathbf{x}_i$，则可以发现,

$$
\mathbf{d}_i^\top\mathbf{e}_{i+1}=0
$$

共轭梯度下降法源于二次规划问题的求解，即

$$
\min\limits_{\mathbf{x}}\quad \frac12\mathbf{x}^\top\mathbf{A}\mathbf{x}-\mathbf{b}^\top\mathbf{x}\quad(\mathbf{A}\succeq0)
$$

其梯度为$\nabla f(\mathbf{x})=\mathbf{Ax}-b\triangleq r(\mathbf{x})$，则求解最优值$\mathbf{x}^*$等价于求解方程组$\mathbf{Ax}-\mathbf{b}=\mathbf{0}$。如果$\mathbf{A}$是一个对称正定矩阵，那么必然可以构建一个$\mathbb{R}^n$空间的一个基，显然基的每个向量与其它基向量是共轭的。

下降方向能不能和这些基向量建立联系呢？答案是肯定的。易知，最优解$\mathbf{x}^\top$可以表示为

$$
\mathbf{x}^*=\sum_{i=0}^{n-1}\alpha_i\mathbf{d}_i\tag{12}
$$

如果$\alpha_i,\mathbf{d}_i$都已知，则$\mathbf{x}^*$可通过上式确定。

##### $\mathbf{A}$-共轭

**定义1**.假设$\mathbf{A}$是一个对称正定矩阵，那么称向量$\mathbf{d}_i,\mathbf{d}_j$是$\mathbf{A}$-共轭的，如果满足，

$$
\mathbf{d}_i^\top\mathbf{A}\mathbf{d}_j=0,\quad i\neq j.
$$

**定理1**. 两两向量相互$\mathbf{A}$-共轭的向量集$\{\mathbf{d}_0,...,\mathbf{d}_{n-1}\}$构成了一个$\mathbb{R}^n$空间的一个基，即$\{\mathbf{d}_0,...,\mathbf{d}_{n-1}\}$线性无关。

有了$\mathbf{A}$-共轭就可以来确定式(12)的各参数值了。

- 首先求$\alpha_i$的表达式。

对式(12)左右同时乘上$\mathbf{d}_k^\top\mathbf{A}$，利用$\mathbf{A}$-共轭性可得

$$
\begin{split}
\mathbf{d}_k^\top\mathbf{A}\mathbf{x}^*&=\sum_{i=0}^{n-1}\alpha_i\mathbf{d}_k^\top\mathbf{Ad}_i\\
\Rightarrow\alpha_k&=\frac{\mathbf{d}_k^\top\mathbf{b}}{\mathbf{d}_k^\top\mathbf{Ad}_k}
\end{split}
$$

可以看出，$\alpha_k$只与搜索方向$\mathbf{d}_k$有关，因此，只需要迭代$n$次就可以计算出$\mathbf{x}^*$，即

$$
\mathbf{x}^*=\sum_{i=0}^{n-1}\frac{\mathbf{d}_i^\top\mathbf{b}}{\mathbf{d}_i^\top\mathbf{Ad}_i}\mathbf{d}_i
$$

为了演示上述过程在$n$步计算出$\mathbf{x}^*$，引入如下定理。

**定理2**. 假设$\{\mathbf{d}_0,...,\mathbf{d}_{n-1}\}$是$n$个$\mathbf{A}$-共轭的向量，$\mathbf{x}_0$是初使点，令

$$
\begin{split}
\mathbf{x}_{k+1}&=\mathbf{x}_k+\alpha_k\mathbf{d}_k\\
\mathbf{g}_k&=\mathbf{b}-\mathbf{Ax}_k\\
\alpha_k&=\frac{\mathbf{g}_k^\top\mathbf{d}_k}{\mathbf{d}_k^\top\mathbf{A}\mathbf{d}_k}=\frac{(\mathbf{b}-\mathbf{Ax})_k^\top\mathbf{d}_k}{\mathbf{d}_k^\top\mathbf{A}\mathbf{d}_k}
\end{split}
$$

则迭代$n$次后，$\mathbf{x}_n=\mathbf{x}^*$。




