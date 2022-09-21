### 优化问题求解(2)

&emsp;&emsp;对于不含有不等式约束的优化问题，可以先将不等式约束问题转化为等式约束问题再求解。

#### 内点法
&emsp;&emsp;用内点法求解含有不等式约束的凸优化问题，就是用Newton方法求解一系列等式约束问题，或者求解一系列KKT条件的修改形式。

##### 障碍法

&emsp;&emsp;尝试将不等式约束问题转化为等式约束问题，从而使用Newton方法求解。即，原问题

$$
\begin{equation}
\begin{split}
\min \quad &f_0(\mathbf{x})\\
s.t.\quad &f_i(\mathbf{x})\leq 0,i=1,2,...,N\\
&\mathbf{Ax}=\mathbf{b}
\end{split}
\end{equation}
$$

可以转化为，

$$
\begin{equation}
\begin{split}
\min\quad &f_0(\mathbf{x})+\sum_{i=1}^N I_{-}(f_i(\mathbf{x}))\\
s.t.\quad &\mathbf{Ax}=0\\
\end{split}
\end{equation}
$$

其中$I_{-}(u)$是指示性函数，

$$
I_{-}(u)=\left\{\begin{array}{l}0 & u\leq 0,\\ \infty&u>0. \end{array} \right.
$$

这样，我们就可以转化为等式约束的优化问题，但$I_{-}(u)$不可微，因此不能使用需要梯度的下降方法。

+ **对数障碍**

&emsp;&emsp;指示函数不可微，可以使用一个近似的可微函数来代替：

  $$
  \hat{I}_{-}(u)=-(1/t)\log(-u),\quad t>0
  $$

  ![barier function](../img/barier_fun.png)

&emsp;&emsp;观察函数图像，可以发现对数障碍函数是非减函数，且当$u>0$时函数取值为$\infty$。因此，定义对数障碍函数为,

$$
\phi(\mathbf{x})=-\sum_{i=1}^m\log(-f_i(\mathbf{x}))
$$

&emsp;&emsp;对数障碍函数的梯度和Hessian矩阵为，

  $$
  \begin{equation}
  \begin{split}
  \nabla \phi(\mathbf{x})&=\sum_{i=1}^m\frac{1}{-f_i(\mathbf{x})}\nabla f_i(\mathbf{x})\\
  \nabla^2\phi(\mathbf{x})&=\sum_{i=1}^m\frac{1}{f_i(\mathbf{x})^2}\nabla f_i(\mathbf{x})\nabla f_i(\mathbf{x})^\top+\sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})}\nabla^2 f_i(\mathbf{x})
  \end{split}
  \end{equation}
  $$

&emsp;&emsp;用$\hat{I}_{-}$替换$I_{-}$可得到以下近似，

  $$
  \begin{equation}
  \begin{split}
  \min\limits_{\mathbf{x}}\quad &f_0(\mathbf{x})+\sum_{i=1}^m-(1/t)\log(-f_i(\mathbf{x}))\\
  s.t.\quad &\mathbf{A}\mathbf{x}=\mathbf{b}
  \end{split}
  \end{equation}
  $$


&emsp;&emsp;由于$-(1/t)\log(-u)$是$u$的单增凸函数，上式上的目标函数是可微凸函数。假定恰当的闭性条件成立，则可以用Newton方法来求解。上式等价于以下问题，

$$
  \begin{equation}
  \begin{split}
  \min\quad &tf_0(\mathbf{x})+\phi(\mathbf{x})\\
  s.t.\quad &\mathbf{A}\mathbf{x}=\mathbf{b}
  \end{split}
  \end{equation}
$$

+ **Central Path**

&emsp;&emsp;针对不同的$t>0$值，我们定义$\mathbf{x}^*(t)$为相应优化问题的解，那么，Central path就是指所有点$\mathbf{x}^*(t),t>0$的集合，其中的点被称为central points。所有中心路径上的点由以下充要条件所界定：$\mathbf{x}^*(t)$是严格可行的，即满足，

$$
\mathbf{Ax}^*(t)=\mathbf{b},\quad f_i(\mathbf{x}^*(t))<0,\quad i=1,...,m
$$

并且存在$\hat{\nu}\in\mathbb{R}^p$使得（**中心路径条件**）

  $$
  \begin{equation}
  \begin{split}
  0&=t\nabla f_0(\mathbf{x}^*(t))+\nabla \phi(\mathbf{x}^*(t))+A^\top\hat{\nu}\\
  &=t\nabla f_0(\mathbf{x}^*(t))+\sum_{i=1}^m\frac{1}{-f_i(\mathbf{x}^*(t))}\nabla f_i(\mathbf{x}^*(t))+A^T\hat{\nu}
  \end{split}
  \end{equation}
  $$

成立。

+ **中心路径条件的KKT条件解释**。点$x$等于$\mathbf{x}^*(t)$的充要条件是存在$\lambda,\nu$满足

$$
    \begin{equation}
    \begin{split}
    Ax=b,f_i(\mathbf{x})&\leq 0,i=1,...,m\\
    \lambda&\succeq 0\\
    -\lambda_if_i(\mathbf{x})&=1/t,i=1,...,m\\
    \nabla f_0(\mathbf{x})+\sum_{i=1}^m\lambda_i\nabla f_i(\mathbf{x})+A^T\nu&=0\\
    \end{split}
    \end{equation}
$$
    
&emsp;&emsp;KKT条件和中心条件的唯一不同在于$\lambda_if_i(\mathbf{x})=0$被条件$-\lambda_if_i(\mathbf{x})=1/t$所替换。从上式可以导出中心路径的一个重要性质：**每次个中心点产生对偶可行解，因而给出最优值$\mathbf{p}^*$的一个下界**。


&emsp;&emsp;解释：先定义

$$
  \lambda^*(t)=-\frac{1}{tf_i(\mathbf{x}^*(t))},i=1,...,m,\quad \nu^*(t)=\hat{\nu}/t
$$

，那么$\lambda^*(t)$和$\nu^*(t)$是对偶可行解。原式可以表示成，

  $$
  \nabla f_0(\mathbf{x}^*(t))+\sum_{i=1}^m \lambda^*(t)\nabla f_i(\mathbf{x}^*(t))+\mathbf{A}^\top\nu^*(t)
  $$

  可以看出，$\mathbf{x}^*(t)$使$\lambda=\lambda^*(t),\nu=\nu^*(t)$时的Lagrange函数，

  $$
  L(\mathbf{x},\lambda,\nu)=f_0(\mathbf{x})+\sum_{i=1}^m\lambda f_i(\mathbf{x})+\nu^\top(\mathbf{A}\mathbf{x}-b)
  $$

  达到最小，这意味着和$\nu^*(t)$是对偶可行解。因此，对偶函数是有限的，并且，

  $$
  \begin{equation}
  \begin{split}
  g(\lambda^*(t),\nu^*(t))&=f_0(\mathbf{x}^*(t))+\sum_{i=1}^m\lambda_i^*f_i(\mathbf{x})+\nu^*(t)^\top(\mathbf{A}\mathbf{x}^*(t)-b)\\
  &=f_0(\mathbf{x}^*(t))-m/t
  \end{split}
  \end{equation}
  $$

  这表明$\mathbf{x}^*(t)$和对偶可行解$\lambda^*(t),\nu^*(t)$之间的对偶间隙就是$m/t$。作为一个重要的结果，我们有，

  $$
  f_0(\mathbf{x}^*(t))-p^*\leq m/t
  $$

  即$\mathbf{x}^*(t)$是和最优值偏差在$m/t$之内的次优解。也证实了$\mathbf{x}^*(t)$随着$t\rightarrow \infty$而收敛于最优解。

  + **障碍函数方法**
  
  | 算法：障碍法                                               |
  | :----------------------------------------------------------- |
  | <br />1. 给定严格可行点$\mathbf{x}，t:=t^{(0)},\mu>1$，误差阈值$\epsilon>0。$<br />2. 重复进行<br />&emsp;&emsp;2.1 中心点步骤。从$\mathbf{x}$开始，在$\mathbf{Ax}=\mathbf{b}$的约束下极小化$tf_0+\phi(\mathbf{x})$，最终确定$\mathbf{x}^*(t)$（$\color{red}{极小化二次可微凸函数：Newton方法求解}$)。<br />&emsp;&emsp;2.2 改进。$\mathbf{x}:=\mathbf{x}^*(t)$。<br />&emsp;&emsp;2.3 停止准则。如果$m/t<\epsilon$则退出。<br />&emsp;&emsp;2.4 增加$t$。 $t:=\mu t$。<br /> |

&emsp;&emsp;Newton步径$\Delta \mathbf{x}_{nt}$以及相关的对偶变量由以下线性方程确定，

  $$
  \begin{equation}\begin{bmatrix}
  t\nabla^2f_0(\mathbf{x})+\nabla^2\phi(\mathbf{x})&\mathbf{A}^\top\\
  \mathbf{A}&0\\
  \end{bmatrix}
  \begin{bmatrix}
  \Delta \mathbf{x}_{nt}\\
  \nu_{nt}
  \end{bmatrix}
  =-\begin{bmatrix}
  t\nabla f_0(\mathbf{x})+\nabla\phi(\mathbf{x})\\0
  \end{bmatrix}
  \end{equation}
  $$
  
 