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