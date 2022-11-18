### 优化问题求解(2)

&emsp;&emsp;求解约束优化问题的标准方法是将约束优化问题转化为无约束优化问题：主要有3种：Lagrangian乘子法、罚函数法和增广Lagrangian乘子法。


#### Lagrangian乘子法

&emsp;&emsp;考虑等式约束的凸优化问题，

$$
\begin{split}
\min\limits_{\pmb{x}}\quad &f(\pmb{x})\\
\mathrm{s.t.}\quad &\pmb{Ax}=\pmb{b}
\end{split}
$$

Lagrangian乘子法将上式转换为无约束问题，即Lagrangian目标函数为，

$$
L(\pmb{x},\pmb{\lambda})=f(\pmb{x})+\pmb{\lambda}^\top (\pmb{Ax}-\pmb{b})
$$

则原始优化问题的对偶目标函数为，

$$
\begin{split}
g(\pmb{\lambda})&=\inf\limits_{\pmb{x}} L(\pmb{x},\pmb
{\lambda})\\
&=-\sup\limits_{\pmb{x}}\left(-(\pmb{A}^\top\pmb{\lambda})^\top\pmb{x}-f(\pmb{x})\right)-\pmb{b}^\top\pmb{\lambda}\\
&=-f^*(-\pmb{A}^\top\pmb{\lambda})-\pmb{b}^\top\pmb{\lambda}
\end{split}
$$

其中，$f^*(\pmb{y})=\sup\limits_{\pmb{x}}(\pmb{y}^\top\pmb{x}-f(\pmb{x}))$是$f(\pmb{x})$的凸共轭函数，$\pmb{\lambda}$为对偶变量。根据Lagrangian乘子法，原始等式约束极小化问题变为对偶极大化问题，即

$$
\max\limits_{\pmb{\lambda}\in\mathbb{R}^m}\quad g(\pmb{\lambda})=-f^*(-\pmb{A}^\top\pmb{\lambda})-\pmb{b}^\top\pmb{\lambda}
$$

假设强对偶性满足，则原始问题与对偶问题的最做优解相同。此时，原始极小化问题的最优解点$\pmb{x}^*$可由下式计算，

$$
\pmb{x}^*=\arg\min\limits_{\pmb{x}}L(\pmb{x},\pmb{\lambda}^*)
$$

&emsp;&emsp;在对偶上升法中，由两个步骤组成，

$$
\begin{split}
\pmb{x}_{k+1}&=\arg\min\limits_{\pmb{x}}L(\pmb{x},\pmb{\lambda}_k)\\
\pmb{\lambda}_{k+1}&=\pmb{\lambda}_k+\mu_k(\pmb{Ax}_{k+1}-\pmb{b})
\end{split}
$$


#### 增广Lagrangian乘子法

&emsp;&emsp;将罚函数与Lagrangian函数相结合，构造出更合适目标函数的方法称为增广Lagrangian乘子法。下面从等式约束和混合约束两个方面讨论该方法。

##### 等式约束

&emsp;&emsp;考虑等式约束优化问题。记$\pmb{h}(\pmb{x})=[h_1(\pmb{x}),...,h_q(\pmb{x})]^\top$。对Lagrangian目标函数$L(\pmb{x},\pmb{\lambda})$加惩罚函数，即，

$$
\begin{split}
L_\rho (\pmb{x},\pmb{\lambda})&=f_0(\pmb{x})+\pmb{\lambda}^\top \pmb{h}(\pmb{x})+\rho \phi(\pmb{h}(\pmb{x}))\\
&=f_0(\pmb{x})+\sum_{i=1}^q \lambda_i h_i(\pmb{x})+\rho\sum_{i=1}^q \phi(h_i(\pmb{x}))
\end{split}
$$

其中$\rho$为惩罚参数。这种将罚函数与Lagrangian函数相结合，构造出更合适的目标函数的方法称为增广Lagrangian乘子法。

&emsp;&emsp;求解无约束优化问题$\min L_\rho (\pmb{x},\pmb{\lambda})$的对偶上升法由以下两个更新组成，

$$
\begin{split}
\pmb{x}_{k+1}&=\arg\min\limits_{\pmb{x}} L_\rho (\pmb{x},\pmb{\lambda}_k)\\
\pmb{\lambda}_{k+1}&=\pmb{\lambda}_k + \rho_k\nabla_{\pmb{\lambda}} L_\rho (\pmb{x}_{k+1},\pmb{\lambda}_k)
\end{split}
$$

可以看出，

&emsp;&emsp;1. 若$\rho=0$，则退化为标准的Lagrangian乘子法。

&emsp;&emsp;2. 若$\pmb{\lambda}=0$，退化为标准罚函数法。

&emsp;&emsp;**例**：若取$\pmb{h}(\pmb{x})=\pmb{Ax}-\pmb{b}$， $\phi(\pmb{h}(\pmb{x}))=\frac12\lVert \pmb{Ax}-\pmb{b}\rVert_2^2$，则增广函数为，

$$
L_\rho (\pmb{x},\pmb{\lambda})=f_0(\pmb{x})+\pmb{\lambda}^\top (\pmb{Ax}-\pmb{b})+\frac{\rho}{2}\lVert \pmb{Ax}-\pmb{b}\rVert_2^2
$$

对应的对偶上升法更新为，

$$
\begin{split}
\pmb{x}_{k+1}&=\arg\min\limits_{\pmb{x}} L_\rho (\pmb{x},\pmb{\lambda}_k)\\
\pmb{\lambda}_{k+1}&=\pmb{\lambda}_k + \rho_k (\pmb{Ax}_{k+1}-\pmb{b})
\end{split}
$$

##### 混合约束

&emsp;&emsp;考虑不等式和等式约束同时存在的混合约束问题，

$$
\begin{split}
\min\limits_{\pmb{x}} \quad &f(\pmb{x})\\
\mathrm{s.t.}\quad &\pmb{Ax}=\pmb{b}\\
 &\pmb{Bx}\preceq \pmb{h}
\end{split}
$$

令非负向量$\pmb{s}\succeq 0$为松驰变量，使得$\pmb{Bx}+\pmb{s}=\pmb{h}$；以及罚函数$\phi(\pmb{g}(\pmb{x}))=\frac12 \lVert \pmb{g}(\pmb{x})\rVert_2^2$，则增广目标函数为，

$$
\begin{split}
L_\rho (\pmb{x,s,\lambda,\nu})&=f(\pmb{x})+\pmb{\lambda}^\top (\pmb{Ax}-\pmb{b})+\pmb{\nu}^\top(\pmb{Bx}+\pmb{s}-\pmb{h})\\
&+\frac{\rho}{2}(\lVert \pmb{Ax-b}\rVert_2^2+\lVert \pmb{Bx}+\pmb{s}-\pmb{h}\rVert_2^2)
\end{split}
$$

对应的对偶上升法更新为，

$$
\begin{split}
\pmb{x}_{k+1}&=\arg\min\limits_{\pmb{x}}L_\rho (\pmb{x},\pmb{s}_k,\pmb{\lambda}_k,\pmb{\nu}_k)\\
\pmb{s}_{k+1}&=\arg\min\limits_{\pmb{x}}L_\rho (\pmb{x}_{k+1},\pmb{s},\pmb{\lambda}_k,\pmb{\nu}_k)\\
\pmb{\lambda}_{k+1}&=\pmb{\lambda}_k + \rho_k (\pmb{Ax}_{k+1}-\pmb{b})\\
\pmb{\nu}_{k+1}&=\pmb{\nu}_k + \rho_k (\pmb{Bx}_{k+1}+\pmb{s}_{k+1}-\pmb{h})
\end{split}
$$



#### 内点法
&emsp;&emsp;用内点法求解含有不等式约束的凸优化问题，就是用Newton方法求解一系列等式约束问题，或者求解一系列KKT条件的修改形式。

##### 障碍法

&emsp;&emsp;尝试将不等式约束问题转化为等式约束问题，从而使用Newton方法求解。即，原问题

$$
\begin{equation}
\begin{split}
\min \quad &f_0(\pmb{x})\\
s.t.\quad &f_i(\pmb{x})\leq 0,i=1,2,...,N\\
&\pmb{Ax}=\pmb{b}
\end{split}
\end{equation}
$$

可以转化为，

$$
\begin{equation}
\begin{split}
\min\quad &f_0(\pmb{x})+\sum_{i=1}^N I_{-}(f_i(\pmb{x}))\\
s.t.\quad &\pmb{Ax}=0\\
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
\phi(\pmb{x})=-\sum_{i=1}^m\log(-f_i(\pmb{x}))
$$

&emsp;&emsp;对数障碍函数的梯度和Hessian矩阵为，

  $$
  \begin{equation}
  \begin{split}
  \nabla \phi(\pmb{x})&=\sum_{i=1}^m\frac{1}{-f_i(\pmb{x})}\nabla f_i(\pmb{x})\\
  \nabla^2\phi(\pmb{x})&=\sum_{i=1}^m\frac{1}{f_i(\pmb{x})^2}\nabla f_i(\pmb{x})\nabla f_i(\pmb{x})^\top+\sum_{i=1}^m \frac{1}{-f_i(\pmb{x})}\nabla^2 f_i(\pmb{x})
  \end{split}
  \end{equation}
  $$

&emsp;&emsp;用$\hat{I}_{-}$替换$I_{-}$可得到以下近似，

  $$
  \begin{equation}
  \begin{split}
  \min\limits_{\pmb{x}}\quad &f_0(\pmb{x})+\sum_{i=1}^m-(1/t)\log(-f_i(\pmb{x}))\\
  s.t.\quad &\pmb{A}\pmb{x}=\pmb{b}
  \end{split}
  \end{equation}
  $$


&emsp;&emsp;由于$-(1/t)\log(-u)$是$u$的单增凸函数，上式上的目标函数是可微凸函数。假定恰当的闭性条件成立，则可以用Newton方法来求解。上式等价于以下问题，

$$
  \begin{equation}
  \begin{split}
  \min\quad &tf_0(\pmb{x})+\phi(\pmb{x})\\
  s.t.\quad &\pmb{A}\pmb{x}=\pmb{b}
  \end{split}
  \end{equation}
$$

+ **Central Path**

&emsp;&emsp;针对不同的$t>0$值，我们定义$\pmb{x}^*(t)$为相应优化问题的解，那么，Central path就是指所有点$\pmb{x}^*(t),t>0$的集合，其中的点被称为central points。所有中心路径上的点由以下充要条件所界定：$\pmb{x}^*(t)$是严格可行的，即满足，

$$
\pmb{Ax}^*(t)=\pmb{b},\quad f_i(\pmb{x}^*(t))<0,\quad i=1,...,m
$$

并且存在$\hat{\nu}\in\mathbb{R}^p$使得（**中心路径条件**）

  $$
  \begin{equation}
  \begin{split}
  0&=t\nabla f_0(\pmb{x}^*(t))+\nabla \phi(\pmb{x}^*(t))+A^\top\hat{\nu}\\
  &=t\nabla f_0(\pmb{x}^*(t))+\sum_{i=1}^m\frac{1}{-f_i(\pmb{x}^*(t))}\nabla f_i(\pmb{x}^*(t))+A^T\hat{\nu}
  \end{split}
  \end{equation}
  $$

成立。

+ **中心路径条件的KKT条件解释**。点$x$等于$\pmb{x}^*(t)$的充要条件是存在$\lambda,\nu$满足

$$
    \begin{equation}
    \begin{split}
    Ax=b,f_i(\pmb{x})&\leq 0,i=1,...,m\\
    \lambda&\succeq 0\\
    -\lambda_if_i(\pmb{x})&=1/t,i=1,...,m\\
    \nabla f_0(\pmb{x})+\sum_{i=1}^m\lambda_i\nabla f_i(\pmb{x})+A^T\nu&=0\\
    \end{split}
    \end{equation}
$$
    
&emsp;&emsp;KKT条件和中心条件的唯一不同在于$\lambda_if_i(\pmb{x})=0$被条件$-\lambda_if_i(\pmb{x})=1/t$所替换。从上式可以导出中心路径的一个重要性质：**每次个中心点产生对偶可行解，因而给出最优值$\pmb{p}^*$的一个下界**。


&emsp;&emsp;解释：先定义

$$
  \lambda^*(t)=-\frac{1}{tf_i(\pmb{x}^*(t))},i=1,...,m,\quad \nu^*(t)=\hat{\nu}/t
$$

，那么$\lambda^*(t)$和$\nu^*(t)$是对偶可行解。原式可以表示成，

  $$
  \nabla f_0(\pmb{x}^*(t))+\sum_{i=1}^m \lambda^*(t)\nabla f_i(\pmb{x}^*(t))+\pmb{A}^\top\nu^*(t)
  $$

  可以看出，$\pmb{x}^*(t)$使$\lambda=\lambda^*(t),\nu=\nu^*(t)$时的Lagrange函数，

  $$
  L(\pmb{x},\lambda,\nu)=f_0(\pmb{x})+\sum_{i=1}^m\lambda f_i(\pmb{x})+\nu^\top(\pmb{A}\pmb{x}-b)
  $$

  达到最小，这意味着和$\nu^*(t)$是对偶可行解。因此，对偶函数是有限的，并且，

  $$
  \begin{equation}
  \begin{split}
  g(\lambda^*(t),\nu^*(t))&=f_0(\pmb{x}^*(t))+\sum_{i=1}^m\lambda_i^*f_i(\pmb{x})+\nu^*(t)^\top(\pmb{A}\pmb{x}^*(t)-b)\\
  &=f_0(\pmb{x}^*(t))-m/t
  \end{split}
  \end{equation}
  $$

  这表明$\pmb{x}^*(t)$和对偶可行解$\lambda^*(t),\nu^*(t)$之间的对偶间隙就是$m/t$。作为一个重要的结果，我们有，

  $$
  f_0(\pmb{x}^*(t))-p^*\leq m/t
  $$

  即$\pmb{x}^*(t)$是和最优值偏差在$m/t$之内的次优解。也证实了$\pmb{x}^*(t)$随着$t\rightarrow \infty$而收敛于最优解。

  + **障碍函数方法**
  
  | 算法：障碍法                                               |
  | :----------------------------------------------------------- |
  | <br />1. 给定严格可行点$\pmb{x}，t:=t^{(0)},\mu>1$，误差阈值$\epsilon>0。$<br />2. 重复进行<br />&emsp;&emsp;2.1 中心点步骤。从$\pmb{x}$开始，在$\pmb{Ax}=\pmb{b}$的约束下极小化$tf_0+\phi(\pmb{x})$，最终确定$\pmb{x}^*(t)$（$\color{red}{极小化二次可微凸函数：Newton方法求解}$)。<br />&emsp;&emsp;2.2 改进。$\pmb{x}:=\pmb{x}^*(t)$。<br />&emsp;&emsp;2.3 停止准则。如果$m/t<\epsilon$则退出。<br />&emsp;&emsp;2.4 增加$t$。 $t:=\mu t$。<br /> |

&emsp;&emsp;Newton步径$\Delta \pmb{x}_{nt}$以及相关的对偶变量由以下线性方程确定，

  $$
  \begin{equation}\begin{bmatrix}
  t\nabla^2f_0(\pmb{x})+\nabla^2\phi(\pmb{x})&\pmb{A}^\top\\
  \pmb{A}&0\\
  \end{bmatrix}
  \begin{bmatrix}
  \Delta \pmb{x}_{nt}\\
  \nu_{nt}
  \end{bmatrix}
  =-\begin{bmatrix}
  t\nabla f_0(\pmb{x})+\nabla\phi(\pmb{x})\\0
  \end{bmatrix}
  \end{equation}
  $$
  
 ##### **原对偶内点法**

 &emsp; &emsp;原对偶内点法和障碍方法非常相似，但也有一些差别。
  + 仅有一层迭代，没有障碍方法的内部迭代和外部迭代的区分。每次迭代时同时更新原对偶变量。
  + 通过将Newton方法应用于修改的KKT方程（即对障碍中心点问题的最优性条件）确定原对偶内点法的搜索方向。原对偶搜索方向和障碍方法导出的搜索方向相似，但不完全相同。
  + 在原对偶内点法中，原对偶迭代值不需要是可行的。

 &emsp; &emsp;原对偶方法经常比障碍方法有效，特别是高精度场合，因为它们可以展现超线性收敛性质。原对偶内点法相对于障碍方法所具有的另一个优点是，它们可以有效处理可行但不严格可行的问题。

###### **原对偶搜索方向**

&emsp;&emsp;如同障碍方法，我们从修改KKT条件开始，该条件可以表述为$r_t(\pmb{x},\lambda,\nu)=0, t>0$，其中

$$
r_t(\pmb{x},\pmb{\lambda,\nu})=\begin{bmatrix} \nabla f_0(\pmb{x})+D\pmb{f}(\pmb{x})^\top\pmb{\lambda}+\pmb{A}^\top\pmb{\nu}\\-\pmb{diag}(\pmb{\lambda})\pmb{f}(\pmb{x})-(1/t)\pmb{1}\\\pmb{Ax-b} \end{bmatrix}
$$

此处的$\pmb{f}:\mathbb{R}^n\rightarrow\mathbb{R}^m$和它的导数矩阵$D\pmb{f}$由下式给出，

$$
\pmb{f}(\pmb{x})=\begin{bmatrix} f_1(\pmb{x})\\ \vdots\\f_m(\pmb{x})\end{bmatrix},\quad D\pmb{f}=\begin{bmatrix} \nabla f_1(\pmb{x})^\top\\ \vdots\\\nabla f_m(\pmb{x})^\top\end{bmatrix}
$$

&emsp;&emsp;如果$\pmb{x,\lambda,\nu}$满足$r_t(\pmb{x,\lambda,\nu})=0$（且$f_i(\pmb{x})<0)$，则$\pmb{x}=\pmb{x}^*(t),\lambda=\lambda^*(t),\nu=\nu^*(t)$。特别地$\pmb{x}$是**原可行的**，$\lambda,\nu$是**对偶可行的**，对偶间隙为$m/t$。

&emsp;&emsp;我们将$r_t$的成分命名为如下：

1. **对偶残差**

$$
r_{dual}=\nabla f_0(\pmb{x})+D\pmb{f}(\pmb{x})^\top\lambda+\pmb{A}^\top\nu
$$

2. **原残差**

$$
r_{pri}=\pmb{Ax-b}
$$

3. **中心残差**(修改的互补性条件)

$$
r_{cent}=-\pmb{diag}(\lambda)\pmb{f}(\pmb{x})-(1/t)\pmb{1}
$$

+ 先固定$t$，考虑从满足$\pmb{f}(\pmb{x})\prec 0,\lambda \succ0$的点$(\pmb{x},\lambda,\nu)$开始求解非线性方程$r_t(\pmb{x},\lambda,\nu)=0$的Newton步径。将当前点和Newton步径分别记为，

  $$
  \pmb{y}=(\pmb{x},\lambda,\nu),\quad \Delta \pmb{y}=(\Delta \pmb{x},\Delta\lambda,\Delta \nu)
  $$

  决定Newton步径的线性方程为，

  $$
  r_t(\pmb{y}+\Delta \pmb{y})\approx r_t(\pmb{y})+Dr_t(\pmb{y})\Delta \pmb{y}=0
  $$

  即，$\Delta \pmb{y}=-Dr_t(\pmb{y})^{-1}r_t(\pmb{y})$。于是我们有，

  $$
  \begin{bmatrix} \nabla^2 f_0(\pmb{x})+\sum_{i=1}^m\lambda_i\nabla^2f_i(\pmb{x})&D\pmb{f}(\pmb{x})^\top&\pmb{A}^\top\\
  -\pmb{diag}(\lambda)D\pmb{f}(\pmb{x})&-\pmb{diag}(\pmb{f}(\pmb{x}))&0\\
  \pmb{A}&0&0
  \end{bmatrix}
  \begin{bmatrix} 
  \Delta \pmb{x}\\ \Delta\lambda\\ \Delta\nu
  \end{bmatrix}
  =-\begin{bmatrix}
  r_{dual}\\r_{cent}\\r_{pri}
  \end{bmatrix}
  $$

  所谓**原对偶搜索方向**$\Delta y_{pd}=(\Delta x_{pd},\Delta\lambda_{pd},\Delta\nu_{pd})$就是上式的解。

###### **代理对偶间隙**

&emsp;&emsp;在原-对偶内点法中，迭代点$\pmb{x}_k,\lambda_k,\nu_k$不一定是可行解(not necessarity feasitble),除了在算法收敛的极限情况。这意味着我们不能方便的在每个步骤$k$计算对偶间隙$\eta_k$。因此，可以定义一个代理对偶间隙(surrogate duality gap)如下，

$$
\hat{\eta}(\pmb{x},\lambda)=-\pmb{f}(\pmb{x})^\top\lambda
$$

该代理间隙有可能成为对偶间隙，当$\pmb{x}$是原问题可行且$\lambda,\nu$是对偶可行时（即，$r_{\mathrm{pri}}=0 \wedge r_{\mathrm{dual}}=0$）。注意：参数$t$对应的代理间隙为$\hat{\eta}=m/\hat{\eta}$.



+ **原-对偶方法**
  
  | 算法：原-对偶方法                                               |
  | :----------------------------------------------------------- |
  | <br />1. 给定满足$f_1(\pmb{x})<0,\cdots f_m(\pmb{x})<0，\lambda\succ 0, \mu>1$，误差阈值$\epsilon_{\mathrm{feas}}>0, \epsilon>0。$<br />2. 重复进行<br />&emsp;&emsp;2.1 设置$t$。$t:= \mu m/\hat{\eta}.$<br />&emsp;&emsp;2.2 计算原-对偶方法$\Delta\pmb{y}_{pd}$<br />&emsp;&emsp;2.3 线性搜索并更新。求得步长$s>0$，并设置$\pmb{y}=\pmb{y}+s\Delta\pmb{y}_{\mathrm{pd}}$<br />3 until ($\lVert r_{\mathrm{pri}}\rVert_2\le \epsilon_{\mathrm{feas}}, \lVert r_{\mathrm{dual}}\rVert_2\le \epsilon_{\mathrm{feas}}$ and $\hat{\eta}\le\epsilon$).<br /> |
  
