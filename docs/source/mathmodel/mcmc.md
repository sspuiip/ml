# 随机模拟

&emsp;&emsp;随机模拟方法的**主要思想**是随机生成样本，利用这些生成的样本完成近似计算，也称为蒙特卡罗方法(Monte Carlo Simulation)。

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

```python
from pylab import *
from numpy import *

def qsample():
    return random.rand()*4.0

def p(x):
    return 0.3*exp(-(x-0.3)**2)+0.7*exp(-(x-2.0)**2/0.3)


def rejection(n):
    M = 0.72
    samples = zeros(n,dtype=float)
    count=0
    for i in range(n):
        accept = False
        while not accept:
            x = qsample()
            u = random.rand()*M
            if u<p(x):
                accept = True
                samples[i]=x
            else:
                count += 1
    print("count",count)
    return samples

if __name__=="__main__":
    
    x = arange(0,4,0.01)
    x2= arange(-0.5,4.5,0.1)
    realdata = 0.3*exp(-(x-0.3)**2) + 0.7* exp(-(x-2.)**2/0.3)
    box = ones(len(x2))*0.75#0.8
    box[:5] = 0
    box[-5:] = 0
    plot(x,realdata,'k',lw=1)
    plot(x2,box,'k--',lw=1)
    import time
    t0=time.time()
    samples = rejection(10000)
    t1=time.time()
    print("Time ",t1-t0)
    
    hist(samples,100,density=1,fc='y')
    xlabel('x',fontsize=24)
    ylabel('p(x)',fontsize=24)
    axis([-0.5,4.5,0,1])
    show()
```

## 重要性采样

&emsp;&emsp;重要性采样(Importance sampling)和接受-拒绝采样相似，也是借助易采样分布$q(x)$来间接采样$p(x)$的样本。

$$
\begin{split}
\mathbb{E}[f]&=\int f(x)p(x)dx\\
&=\int f(x)\frac{p(x)}{q(x)}q(x)dx\\
&\approx \frac{1}{N}\sum_{i=1}^N \underbrace{\frac{p(x^{(i)})}{q(x^{(i)})}}_{\mathrm{importance\ weight}}f(x^{(i)}),\quad x^{(i)}\sim q(x)
\end{split}
$$

## MCMC

&emsp;&emsp;与前面的采样方法不同，MCMC算法的**主要思想**是：通过Markov Chain进行状态转移（只需要一个状态转移矩阵），当Markov Chain收敛后(可以证明满足detail平衡条件的马氏链具有稳态分布)，到达所有状态$\pmb{x}_i$的概率正好是某个概率分布$\pmb{x}_i\sim p(\pmb{x})$。

### 马氏链

&emsp;&emsp;简单地说，马氏链就是一个下一状态只与当前状态有关的随机过程，即，

$$
P[X_{n+1}=j|X_n=i,X_{n-1}=i_{n-1},...,X_0=i_0]=P[X_{n+1}=j|X_n=i]
$$

&emsp;&emsp;**例1**. 社会学家经常把人按其经济状态分为3类：上层、中层、下层。决定一个人的收入阶层最重要的因素就是其父母的收入阶层。如果父母的收入属于下层类别，那么他的孩子属于下层收入的概率是0.65，属于中层的概率是0.28，属于上层收入的概率是0.07。事实上，从父代到子代，收入阶层的变化转移概率大致如下，

$$
\pmb{P}=\begin{bmatrix}0.65&0.28&0.07\\ 0.15&0.67&0.18\\ 0.12&0.36&0.52 \end{bmatrix}
$$

假设当前这一代人处在下、中、上层的比例是概率分布向量$\pi_0=[\pi_0(1),\pi_0(2),\pi_0(3)]$，那么子代处于三层的比例分布为$\pi_1=\pi_0\pmb{P}$，后代的比例分布则是$\pi_n=\pi_{n-1}\pmb{P}=...=\pi_0\pmb{P}^n$。若有初始概率$\pi_0=[0.21,0.68,0.11]$，则可以计算前$n$代人的比例分布最终稳定在$\pi=[0.287, 0.488, 0.225]$，且这一稳定分布与初始概率无关。

```python
import numpy as np
np.set_printoptions(precision=3)

# Markov chain demo

#状态转移矩阵
def PrMatrix():
    return np.array([[0.65,0.28,0.07],
                     [0.15,0.67,0.18],
                     [0.12,0.36,0.52]]) 
#迭代with Pi0
def iterN(Pi0,P,n):
    pi=Pi0
    for i in range(n):
        print('pi',i,':',pi)
        pi=np.dot(pi,P)
        
#概率转移矩阵稳定收敛
def calcPn(P,n):
    Pn=P
    for i in range(n):
        print('P',Pn)
        Pn=np.dot(Pn,P)

if __name__=='__main__':
    P=PrMatrix()
    pi=np.array([0.21,0.68,0.11])    
    iterN(pi, P, 25)
    
    pi=np.array([0.75,0.15,0.1])    
    iterN(pi, P, 25)
```

&emsp;&emsp;收敛行为主要由概率转移矩阵$P$决定，可以计算一下$P^n$

$$
P^{21}=...=P^{100}=...=\begin{bmatrix}0.287& 0.488& 0.225\\ 0.287& 0.488& 0.225\\ 0.287& 0.488& 0.225\\\end{bmatrix}
$$

当$n$足够大时，$P^n$矩阵的每一行都是稳定地收敛到$\pi=[0.287, 0.488, 0.225]$这个分布。那么满足什么样的马氏链才能具有这一特征呢？

>**定理（马氏链收敛）**. 如果一个非周期马氏链具有转移概率矩阵$P$，且它的任意两个状态是连通，那么$\lim_{n\rightarrow \infty}P_{ij}^n$存在且与$i$无关，记$\lim_{n\rightarrow \infty}P_{ij}^n=\pi(j)$，则有
>
>1.   

$$
\lim_{n\rightarrow \infty}P_{ij}^n=\begin{bmatrix}\pi(1)&\pi(2)&...&\pi(j)&...\\ \pi(1)&\pi(2)&...&\pi(j)&...\\ ...&...&...&...&...\\ \pi(1)&\pi(2)&...&\pi(j)&...\\ ...&...&...&...&...\\ \end{bmatrix}
$$

>2.  $\pi(j)=\sum_{i=1}^\infty \pi(i)P_{ij}$；
>3. $\pi$是方程$\pi=\pi P$的唯一非负解。且$\sum_{i=0}^\infty \pi_i=1$。
>$\pi$则被称为马氏链的平稳分布。

### Markov Chain Monte Carlo

&emsp;&emsp;由于马氏链能收敛到平稳分布$p(x)$，如果我们能构造一个转移矩阵为$P$的马氏链，使得该马氏链的平稳分布正好是所需的$p(x)$，那么我们就可以从任何一个初始状态$x_0$出发，沿着马氏链转移，得到一个转移序列$x_0,x_1,...,x_n,x_{n+1},...$，只要马氏链在某一个$n$处收敛，则后续状态序列就是平稳分布$\pi=p(x)$的采样样本$x_n,x_{n+1},....$。

&emsp;&emsp;马氏链的收敛性主要由转移矩阵决定，那么基于马氏链做采样的关键问题就是如何构造转移矩阵$P$，并达到平稳分布恰好是所需的分布$p(x)$。以下定理给出了具体的条件。

>**定理（细致平稳条件）**. 如果非周期马氏链的转移矩阵$P$和分布$\pi(x)$满足，

$$
\pi(i)P_{ij}=\pi(j)P_{ji},\quad \forall i,j
\tag{1}
$$

>则$\pi(x)$是马氏链的平稳分布，该式也称为细致平稳条件(detailed balance condition)。

<!-- &emsp;&emsp;马氏链的收敛性主要由转移矩阵决定，基于马氏链采样的关键问题是如何构造转移矩阵，使得平稳分布是我们所需的分布。 -->

#### Metropolis-Hasting采样算法

&emsp;&emsp;假设我们已经有了转移矩阵$Q$，用$q(j|i)$表示从状态$i$转移到状态$j$的概率，通常情况下，

$$
p(i)q(j|i)\neq p(j)q(i|j)
$$

不满足细致平稳条件，所以$p(x)$不是这个马氏链的平稳分布。那么，能否改造一下使得条件满足呢？引入接受率$\alpha(j|i)$，使得下式成立，

$$
p(i)q(j|i)\alpha(j|i)= p(j)q(i|j)\alpha(i|j)
$$

显然，令$\alpha(j|i)=p(j)q(i|j), \alpha(i|j)=p(i)q(j|i)$，等式成立，即

$$
p(i)\underbrace{q(j|i)\alpha(j|i)}_{q'(j|i)}=p(j)\underbrace{q(i|j)\alpha(i|j)}_{q'(i|j)}
$$

$\alpha(j|i)$的物理意义：状态$i$跳转到状态$j$时，以$\alpha(j|i)$的概率接受这个跳转。最终把原转移矩阵$Q$的一个马氏链改造成了具有转移矩阵$Q'$的马氏链，而$Q'$恰好满足细致平稳条件，因此马氏链$Q'$的平稳分布就是$p(x)$。把上述过程整理一下，就可以得到从$p(x)$分布采样的MCMC算法。

| MCMC采样算法 |
| :--- |
|1. 初使化马氏链初使状态$X_0=x_0$;<br>|
|2. while(i < 最大迭代次数){<br>&emsp;&emsp;$X_i=x_i$，采样一个样本$y\sim q(x\vert x_t)$；<br> &emsp;&emsp;从均匀分布采样$u\sim \textrm{Uniform}[0,1]$；<br>&emsp;&emsp;如果$u<\alpha(y\vert x_t)=p(y)q(x_t\vert y)$则接受转移$X_{t+1}=y$；<br>&emsp;&emsp;否则不接受转移即$X_{t+1}=x_t$；<br>}|

&emsp;&emsp;上述算法中，接受率$\alpha(j|i)$可能会偏小会导致马氏链拒绝大量的跳转，也就是到达收敛状态速度会太慢。假设$\alpha(j|i)=0.1, \alpha(i|j)=0.2$时满足细致平稳条件，即

$$
p(i)q(j|i)\times 0.1 = p(j)q(i|j)\times 0.2
$$

上式左右两边同时扩大5倍，可以改写成

$$
p(i)q(j|i)\times 0.5 = p(j)q(i|j)\times 1
$$

可以看出，接受率提高了5倍，而细致平稳条件仍满足。这使得我们可以对接受率同比例放大，使得两个数中最大的一个先达到1，这样我们就提高了采样中的跳转接受率。所以可以考虑接受率，

$$
\alpha(j|i)\triangleq\frac{\alpha(j|i)*K}{\alpha(i|j)*K}=\min\left\{ \frac{p(j)q(i|j)}{p(i)q(j|i)},1\right\}\tag{2}
$$

&emsp;&emsp;经过接受率放大后的MCMC算法就是所谓的**Metropolis-Hasting算法**。

| Metropolis-Hasting采样算法 |
| :--- |
|1. 初使化马氏链初使状态$X_0=x_0$;<br>|
|2. while(i < 最大迭代次数){<br>&emsp;&emsp;$X_i=x_i$，采样一个样本$y\sim q(x\vert x_t)$；<br> &emsp;&emsp;从均匀分布采样$u\sim \textrm{Uniform}[0,1]$；<br>&emsp;&emsp;如果$u<\alpha(y\vert x_t)=\min\left\{ \frac{p(j)q(i\vert j)}{p(i)q(j\vert i)},1\right\}$则接受转移$X_{t+1}=y$；<br>&emsp;&emsp;否则不接受转移即$X_{t+1}=x_t$；<br>}|


#### Gibbs采样算法

&emsp;&emsp;对于上述接受率，一般都存在$\alpha<1$的情况，能否找到一个转移矩阵，使得接受率$\alpha=1$成立呢？假设有一个二维概率分布$p(x,y)$，考查$x$坐标相同的两个点$A(x_1,y_1),B(x_1,y_2)$，我们发现有以下等式成立，

$$
\begin{split}
p(x_1,y_1)p(y_2|x_1)&=p(x_1)p(y_1|x_1)p(y_2|x_1)\\
p(x_1,y_2)p(y_1|x_1)&=p(x_1)p(y_2|x_1)p(y_1|x_1)
\end{split}
$$

显然上述所有等式的右边都是相等的，因此，可得，

$$
p(x_1,y_1)p(y_2|x_1)=p(x_1,y_2)p(y_1|x_1),\quad \textrm{i.e.},\quad p(A)p(y_2|x_1)=p(B)p(y_1|x_1)
$$

基于上述等式，对于$x=x_1$这个平行于$y$轴的直线上，如果使用条件分布$p(y|x_1)$做为直线上任意点之间的转移概率，那么该转移概率满足细致平稳条件。同理，对于$y=y_1$这条直线也成立。因此可以构造平面上任意两点之间的转移概率矩阵$Q$,

| 跳转概率 | 条件 |
|:--- | :--- |
|$Q(B\|A)=p(y_B\|x_1)$ | $x_A=x_B=x_1$ |
|$Q(C\|A)=p(x_c\|y_1)$ | $y_A=y_C=y_1$ |
|$Q(D\|A)=0$ | 其它 |

对于上述转移矩阵，很容易验证对于平面上任意两点$X,Y$，满足细致平稳条件，

$$
p(X)Q(Y|X)=p(Y)Q(X|Y)
$$

&emsp;&emsp;采用上述转移矩阵的MCMC采样算法就是所谓的**Gibbs Sampling算法**。

| $n$维Gibbs Sampling采样算法 |
| :--- |
|1. 随机初使化$\pmb{x}^{i}=\{x_j:j=1,...,n$\}|
|2. while(i < 最大迭代次数){<br>&emsp;&emsp;$x^{(i)}_1=p(x_1\|x^{(i)}_2,...,x^{(i)}_n)$<br> &emsp;&emsp;$x^{(i)}_2=p(x_2\|x^{(i+1)}_1,x^{(i)}_3,...,x^{(i)}_n)$；<br>&emsp;&emsp;...<br>&emsp;&emsp;$x^{(i)}_n=p(x_n\|x^{(i+1)}_1,x^{(i+1)}_2,...,x^{(i+1)}_{n-1})$；<br>}|
