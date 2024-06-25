# Random Fourier features

&emsp;&emsp;概率分布Pdf的Fourier变换可表示静态核。Fourier变换可由Monte Carlo近似。Monte Carlo抽取的样本可
称为Random Fourier features, RFF，可以做为贝叶斯回归模型的特征。贝叶斯线性回归的代价$O\min(N^3,M^3)$，$M$为回归模型的特征数量。
当$M<N$时，可以显著减少算法代价。





## 附录

### 复数

复数
: 复数由实部和虚部组成。如$c=a+ib$。$i$为虚数单位且$i^2=-1$。

复数共轭
: -实部相同虚部相反的两个复数共轭。如：$c=a+ib$的共轭复数为$\bar{c}=a-ib$。
: -共轭复数相乘结果为实数。如：$c\bar{c}=a^2+b^2$。
: -$\text{Re}(c)=\frac12(c+\bar{c})$，$\text{img}(c)=\frac{1}{2i}(c-\bar{c})$。

复数运算
: 加法: 实部和虚部分别相加。如$z=a+ib$, $y=c+id$，则$z+y=a+c+i(b+d)$。
: 乘法: 类似于多项式乘法。如$z=a+ib$, $y=c+id$，则$z*y=ac-bd+i(ad+bc)$。



### 欧拉公式

&emsp;&emsp;欧拉公式是指下式：

$$
\boxed{
e^{i\theta}=\cos\theta+i\sin\theta}
$$(eular-formula)

该公式的特例，

$$
\boxed{
e^{i\pi}=-1}
$$(interest-eular)

将$e,\pi,i$三个数学单位统一在一个公式里。

&emsp;&emsp;（一）**Eular公式在指数复平面运算的应用**

&emsp;&emsp;若$c_1,c_2$为两复数，则有以下等式成立，

$$
\begin{split}
e^{c_1+c_2}&=e^{a_1+a_2}\cdot e^{i(b_1+b_2)}\\
&=e^{a_1+a_2}(\cos(b_1+b_2)+i\sin(b_1+b_2))\\
&=e^{a_1+a_2}(\cos b_1\cos b_2-\sin b_1\sin b_2+i(\sin b_1\cos b_2+\cos b_1\sin b_2))\\
&=e^{a_1}(\cos b_1 + i\sin b_1)e^{a_2}(\cos b_2 + i\sin b_2)\\
&=e^{c_1}e^{c_2}
\end{split}
$$(eular-app-1)

因此，乘上$e^{i\theta}$等价于旋转$\theta$角度，即

$$
e^{i\theta_1}e^{i\theta_2}=e^{i(\theta_1+\theta_2)}
$$(eular-app-1-esp)

上式{eq}`eular-app-1-esp`可应用于求三角公式。由式{eq}`eular-formula`可知，其等式右边$\cos\theta+i\sin\theta$为一个复数，即

$$
\begin{split}
\cos(\theta_1+\theta_2)&=\text{Re}(e^{i(\theta_1+\theta_2)})\\
&=\text{Re}(e^{i\theta_1}e^{i\theta_2})\\
&=\text{Re}((\cos\theta_1+i\sin\theta_1)*(\cos\theta_2+i\sin\theta_2))\\
&=\cos\theta_1\cos\theta_2-\sin\theta_1\sin\theta_2
\end{split}
$$(cos-add)

&emsp;&emsp;同理可知，

$$
\begin{split}
\sin(\theta_1+\theta_2)&=\text{img}(e^{i(\theta_1+\theta_2)})\\
&=\sin\theta_1\cos\theta_2+\cos\theta_1\sin\theta_2
\end{split}
$$(sin-add)

&emsp;&emsp;以及，

$$
\begin{split}
\cos(n\theta)+i\sin(n\theta)&=e^{in\theta}\\
&=(e^{i\theta})^n\\
&=(\cos\theta+i\sin\theta)^n
\end{split}
$$(plusn)


