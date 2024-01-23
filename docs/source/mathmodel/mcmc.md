# 随机模拟

&emsp;&emsp;随机模拟方法的主要思想是随机生成样本，利用这些生成的样本完成近似计算，也称为蒙特卡罗方法(Monte Carlo Simulation)。

&emsp;&emsp;一个常见的示例是用随机模拟的方法近拟计算$\pi$的值。

&emsp;&emsp;**例**1. 随机模拟计算$\pi$的值。我们知道单位圆的面积$S=\pi r^2$，如果我们将$1/4$的单位圆置于边长为$1$的正方形中，并在此正方形内均匀地随机的生成一个样本集$\{(x,y)|x,y\sim\mathrm{Uniform}(0,1)\}$，则$\pi$可近似的计算如下：

$$
\begin{split}
\frac{S_{1/4圆}}{S_{正方形}}&\approx\frac{\#\{圆内样本\}}{\#\{正方形内样本\}}\\
\pi&\approx\frac{4*\#\{(x,y)|x^2+y^2 \le 1\}}{\#\{(x,y)\}}
\end{split}
$$

其中$\#\{\}$为集合计数函数。

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Monte Carlo 

if __name__=='__main__':
    n=1000000
    x=stats.uniform.rvs(size=n)
    y=stats.uniform.rvs(size=n)
    zipxy=zip(x,y)
    total = [1 if _[0]**2+_[1]**2 <=1 else 0 for _ in zipxy]
    print('The estimation of pi value is:', 4*sum(total)/len(total))
```
&emsp;&emsp;从上例可以看出，通过$(x,y)$各自在均匀分布$[0,1]$生成样本，通过随机生成的样本计数达到计算$\pi$值的目标。一般来说，均匀分布的样本是相对容易生成。通过线性同余发生器可以生成伪随机数，这些伪随机数序列的各种统计指标和均匀分布是理论计算结果非常接近，可以用于真实的随机数使用。

&emsp;&emsp;**例**2. 计算以下积分，

$$
I=\int_a^b f(x)dx
$$

其中$f(x)$形式复杂，不容易直接积分。对于难积分的函数$f(x)$，常用的方法是利用容易采样的分布生成随机样本，并利用这些样本达到计算积分的目的。**Monte Carlo积分**是一种常见的采样积分方法。通过对上式引入易采样分布$q(x)$，则原式可以写成，

$$
\begin{split}
I&=\int_a^b f(x)dx\\
&=\int_a^b \frac{f(x)}{q(x)}q(x)dx
\end{split}
$$

这样可以通过$q(x)$抽取$n$个样本，计算$\frac{f(x)}{q(x)}$的均值即为原积分$I$的值。以下代码模拟计算$\int_1^3 3x^2dx=26$，取$q(x)$为均匀分布$\mathrm{Uniform}[1,3]$。

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Monte Carlo  f(x)=3x^2 q(x):Uniform[1,3] = 1/(3-1)

def fx(x):
    return 3*x*x;
def qx_pdf(x,a=1,b=3):
    return 1/(b-a);

def integrateFX(x):
    return fx(x)/qx_pdf(x);
    

if __name__=='__main__':
    n=1000
    samples = stats.uniform.rvs(loc=1,scale=2,size=n); #[1,3]均匀抽取样本
    sumFX=sum(integrateFX(samples))
    print("I=",sumFX/n)
```

&emsp;&emsp;**例**3. 计算期望$\mathbb{E}[f(x)]$,

$$
\begin{split}
\mathbb{E}[f(x)]&=\int_{-\infty}^{+\infty} f(x)p(x)dx,\quad x\sim p(x)\\
&\approx \frac{1}{N}\sum_{i=1}^N f(x^{(i)}),\quad x^{(i)}\sim p(x)\\
\end{split}
$$

同理，上式仍然可以使用抽样的方法近似计算。Monte Carlo方法只需要从$p(x)$中采样$x'$，然后对这些样本计算$f(x')$并取均值即可得到期望的近似值。



&emsp;&emsp;随机采样的方法有很多种，比如：逆分布函数法、Box-Muller变换、Monte Carlo、接受-拒绝采样、重要性采样、Markov链等。常见的概率分布，都可以基于均匀分布生成样本.

## Box-Muller变换

>**定理**. 如果随机变量$U_1,U_2\sim \mathrm{Uniform[0,1]}$且相互独立，则有，

$$
\begin{split}
Z_0 &=\sqrt{-2\ln U_1}\cos (2\pi U_2)\\
Z_1 &=\sqrt{-2\ln U_1}\sin (2\pi U_2)\\
\end{split}
$$   

>$Z_0,Z_1$相互独立且服从标准正态分布。

## 接受-拒绝采样

&emsp;&emsp;实际应用中，$p(x)$一般不容易直接采样。此时，我们可以借用一个易采样的分布$q(x)$采样，然后按一定的规则接受和拒绝所有的样本，从而达到近似$p(x)$分布采样的目标。设定一个方便采样的分布$q(x)$，以及一个常量$k$，使得所有的$x$都有$kq(x)>=p(x)$也就是$kq(x)$总在位于$p(x)$的上方。具体采样步骤为，

- 从易采样分布$q(x)$采样$x'$；

- 从均匀分布$[0,kq(x')]$随机采样$u$；

- 如果$u>p(x')$则拒绝样本$x'$，否则接受；

- 重复以上过程，直到采样数满足要求。

需要注意的是，合适的$q$分布比较难找，且不容易确定$k$值。这些问题导致拒绝率升高，无用计算增加。

## 重要性采样

&emsp;&emsp;重要性采样(Importance sampling)和接受-拒绝采样相似，也是借助易采样分布$q(x)$来间接采样$p(x)$的样本。

$$
\begin{split}
\mathbb{E}[f]&=\int f(x)p(x)dx\\
&=\int f(x)\frac{p(x)}{q(x)}q(x)dx\\
&\approx \frac{1}{N}\sum_{i=1}^N \underbrace{\frac{p(x^{(i)})}{q(x^{(i)})}}_{\mathrm{importance\ weight}}f(x^{(i)}),\quad x^{(i)}\sim q(x)
\end{split}
$$