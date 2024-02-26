# 概率图模型

&emsp;&emsp;根据已观察到的数据（样本集）对未知变量（样本所属类别）进行估计，这是从数据学习知识的基本途径，也是机器学习的主要任务。概率模型将这一任务转换为计算变量的概率分布，即利用已知变量推测未知变量，也称之为**推断**。具体来说，假设所关心的变量集为$Y$，可观测变量集为$O$，其它变量为$R$，推断就是通过联合分布$P(Y,R,O)$或条件分布$P(Y,R|O)$计算得到条件分布$P(Y|O)$。其中，联合分布$P(Y,R,O)$称为**生成式模型**，条件分布$P(Y,R|O)$称为**判别式模型**。但是，直接使用概率求和规则消去变量$R$是不可行的，因为即使所有变量只有2种取值的特殊情况，计算复杂度也高达$O(2^{|Y|+|R|})$。

&emsp;&emsp;概率图模型是一种用图来表示变量间关系的概率模型。该图的结点表示一个（组）随机变量，结点之间的边表示变量间的相关关系。根据边的类型不同，图模型又可以继续细分为**有向无环图**（贝叶斯网，Bayesian network）和**无向无环图**（马尔可夫网，Markov network）两种。

## 隐马尔可夫模型

&emsp;&emsp;隐马尔可夫模型（Hidden Markov Model, HMM）是一种有向图模型，主要用于时序数据建模、自然语言处理、语音识别等领域。

```{mermaid}
---
caption: Fig 1. Hidden Markov Model
align: center
---
block-beta
  columns 10
  x1(("x1")) space x2(("x2")) space x3(("x3")) space x4(("...")) space x5(("xn")) space space space space space space space space space space space
  y1(("y1")) space y2(("y2")) space y3(("y3")) space y4(("...")) space y5(("yn"))
  y1 --> y2
  y2 --> y3
  y3 --> y4
  y4 --> y5
  y1 --> x1
  y2 --> x2
  y3 --> x3
  y4 --> x4
  y5 --> x5
```
&emsp;&emsp;如上图所示，隐马尔可夫模型中的变量分为两类，第一类为状态变量$\{y_1,y_2,...,y_n\}$，$y_i$表示第$i$时刻的系统状态，一般状态变量是不可观测的，也称为隐变量；第二类为观测变量$\{x_1,x_2,...,x_n\}$，$x_i$表示为第$i$时刻的观测值。图中的箭头代表的是变量间的依赖关系。具体来说有以下两点：

- $x_t$只依赖$y_t$。$t$时刻的观测值只与$t$时刻的状态相关，与其它状态无关。

- $y_t$只依赖$y_{t-1}$。 该性质也就是所谓的**马尔可夫性**。

根据依赖关系，HMM所有变量的**联合分布**为，

$$
P(x_1,...,x_n,y_1,...,y_n)=P(y_1)P(x_1|y_1)\prod_{i=2}^n P(x_i|y_i)P(y_{i}|y_{i-1})
$$(hmm-joint)

&emsp;&emsp;除了上述变量间的依赖关系（也就是模型的结构信息），确定一个隐马尔可夫模型还需要三组参数，即**模型参数$\lambda=[A,B,\pmb{\pi}]$**：

:::{table} 隐马尔可夫模型的主要参数
:width: 600px
:align: center
:widths: 33,67
| 参数 | 描述 |
| :--- | :--- | 
| **状态转移概率**。<br>记为矩阵$A=[a_{ij}]_{N\times N}$<br>$a_{ij}=P(y_{t+1}=s_j \| y_t=s_i)$ | 各个状态间的跳转概率。<br>在任意时刻$t$，若状态为$s_i$，则下一时刻状态为$s_j$的概率。 |
|**输出观测概率**。<br>记为矩阵$B=[b_{ij}]_{N\times M}$<br>$b_{ij}=P(x_t=o_j \| y_t=s_i)$ | 模型当前状态得到观测值的概率。<br>在任意时刻$t$，若状态为$s_i$，则观测值为$o_j$的概率。 |
|**初使状态概率**。<br>记为$\pmb{\pi}=(\pi_1,...,\pi_N)$<br>$\pi_i = P(y_1=s_i)$ | 模型在初始时刻各状态出现的概率。|
:::

通过上述三组参数可以确定一个隐马尔可夫模型。现实应用中，隐马尔可夫模型一般主要用来解决以下3种问题：

:::{table} 隐马尔可夫模型的主要解决的问题
:width: 600px
:align: center

|应用问题 | 场景 |
| :--- | :---|
|1. 计算观测序列产生概率$P(\pmb{x}\|\lambda)$，也就是如何评估模型与观测序列之间的匹配程度？    | 根据以往观测序列$(x_1,...,x_{n-1}$推测当前时刻观测值$x_n$的可能性，可以转化为求概率$P(\pmb{x}\|\lambda)$。 |
| 2. 给定模型$\lambda$和观测序列$\pmb{x}=(x_1,...,x_n)$，如何找到与此观测序列匹配的隐状态序列$\pmb{y}=(y_1,...,y_n)$，也就是根据观测序列如何推断出隐状态？     | 语音识别任务中，隐藏状态是文字，目标为根据观测信号推断最有可能的状态序列（文字序列）。 |
| 3. 给定观测序列$\pmb{x}=(x_1,...,x_n)$，如何优化参数$\lambda$使得序列出现的概率$P(\pmb{x}\| \lambda)$最大，也就是如何训练模型？| 根据训练样本得到最优参数。根据条件独立性，隐马尔可夫模型的三个问题都能高效求解。 |
:::


## 马尔可夫随机场

&emsp;&emsp;马尔可夫随机场是一种无向图模型。图中结点表示一个（组）变量，结点之间的边表示变量之间的依赖关系。马尔可夫随机场的联合概率分布函数由一组**势函数**（potential functions），也称之为**因子**(factor)，构成。势函数是定义在变量子集上的非负实函数。

&emsp;&emsp;马尔可夫随机场的变量子集根据结点特性可以加以区别。若一个结点子集中任意两结点之间都有边连接，则称该子集为一个**团**（clique）；若在一个团中加入另外任何一个结点后，不再形成团，则该团称为**极大团**（maximal clique）。

### 联合概率

&emsp;&emsp;在马尔可夫随机场中，多个变量之间的联合概率分布可能基于团分解为多个势函数（因子）的乘积，每次个势函数只与一个团关联。具体来说，对于$n$个变量$\pmb{x}=\{x_1,...,x_n\}$，所有团构成的集合为$\mathcal{C}$，团$Q\in\mathcal{C}$相关的变量集合记为$\pmb{x}_Q$，则**联合概率**定义为，

$$
P(\pmb{x})=\frac1Z \prod_{Q\in\mathcal{C}}\psi_Q(\pmb{x}_Q)
$$(markov-joint)

其中，$\psi_Q$为团$Q$对应的势函数，$Z=\sum_{\pmb{x}}\prod_{Q\in\mathcal{C}}\psi_Q(\pmb{x}_Q)$为常数，也称之为规范化因子。实际应用中精确计算$Z$往往很困难，但很多时候并不需要获得$Z$的精确值。

&emsp;&emsp;若变量个数过多，则团的数据会很多，就会对联合概率的计算带来负担。可以发现，只要团$Q$不是极大团，则它必然被一个极大团$Q^*$包含。因此，可以根据极大团来定义联合概率。假设极大团构成的集合为$\mathcal{C}^*$，则有，

$$
P(\pmb{x})=\frac{1}{Z^*} \prod_{Q\in\mathcal{C}^*}\psi_Q(\pmb{x}_Q)
$$(maxclique-joint)

其中，$Z^*=\sum_{\pmb{x}}\prod_{Q\in\mathcal{C}^*}\psi_Q(\pmb{x}_Q)$。


:::{admonition} **例**1. 假设有一随机变量集$\pmb{x}=\{x_1,x_2,...,x_6\}$的马尔可夫随机场如下图所示
:class: dropdown

```{mermaid}
---
caption: Fig 2. 马尔可夫随机场示例 
align: center
---
flowchart LR
  x1((x1)) --- x2((x2))
  x1((x1)) --- x3((x3))
  x2((x2)) --- x4((x4))
  x2((x2)) --- x6((x6))
  x2((x2)) --- x5((x5))
  x5((x5)) --- x6((x6))
  x3((x3)) --- x5((x5))

```
则联合概率分布为，

$$
P(\pmb{x})=\frac1Z \psi_{12}(x_1,x_2)\psi_{13}(x_1,x_3)\psi_{24}(x_2,x_4)\psi_{35}(x_3,x_5)\psi_{256}(x_2,x_5,x_6)
$$

:::

### 条件独立性


&emsp;&emsp;借助分离集得到变量的条件独立性。若结点集$A$到$B$中的结点都必须经过结点集$C$中的结点，则称$C$为$A$,$B$的**分离集**（separating set）。

:::{admonition} **例**2. 图3给出了分离集$C$的示例。子集$B$和$A$的结点相连都要经过子集$C$的结点
:class: dropdown

```{mermaid}
---
caption: Fig 3. 马尔可夫随机场分离集示例 
align: center
---
graph LR   
  x1 -.- x9(("x9"))
  subgraph A 
    x3 --- x1
    x2(("x2")) --- x1(("x1"))
    x2 --- x3(("x3"))    
  end
  x2 --- x4
  x3 --- x5
  subgraph C
    x4(("x4")) --- x5(("x5"))
  end
  x4 --- x6
  x5 --- x7
  subgraph B
    x6(("x6")) --- x7(("x7"))
  end
  x7 -.- x8(("x8"))

```
::: 

- **全局马尔可夫性: 给定两个变量子集的分离集，则这两个变量子集条件独立**

&emsp;&emsp;在例2中，若令$A,B,C$对应的变量子集分别为$\pmb{x}_A,\pmb{x}_B,\pmb{x}_c$，则有$\pmb{x}_A$与$\pmb{x}_C$在给定条件$\pmb{x}_B$的条件下相互独立，即

$$
\pmb{x}_A \bot\pmb{x}_C |\pmb{x}_B
$$

&emsp;&emsp;由全局马尔可夫性可以得到两个有用的结论：

1. **局部马尔可夫性**. 给定某变量的邻接变量，则该变量条件独立于其它变量，$\pmb{x}_v \bot \pmb{x}_{V\backslash \{ n(v)\cup v\}}|\pmb{x}_{n(v)}$。

2. **成对马尔可夫性**. 给定其它所有变量，两个非邻接变量条件独立，$\pmb{x}_u \bot\pmb{x}_v|\pmb{x}_{V\backslash\{u\cup v\}}$。


## 推断与参数学习

&emsp;&emsp;基于概率图模型定义的联合概率分布，我们可以对感兴趣目标变量的**边缘分布进行推断**；或者给定某些观测变量为条件的**条件分布进行推断**。

&emsp;&emsp;对于概率图模型，还需要确定具体分布的参数，这也称之为参数学习（估计）问题。一般使用极大似然估计或最大后验估计方法求解。特别地，如果将待学习的参数视为待推测变量，则参数估计和推断问题就非常相似，可以认为是推断问题。因此，如果没有特别要求，只需要观注概率图模型的推断方法。

&emsp;&emsp;概率图模型的推断方法大致可以分为两类：

1. 精确推断方法. 精确计算目标变量的边缘分布或条件。一般情况下，此类方法随着极大团规模的增长计算复杂度呈指数增长。

2. 近似推断方法. 此类方法期望在较低的时间复杂度获得问题的近似解，实际任务中应用较为广泛。

&emsp;&emsp;具体来说，假设图模型所对应的变量集$\pmb{x}=\{x_1,...,x_n\}$可以拆分为$\pmb{x}_E$和$\pmb{x}_F$两部分，推断问题的目标就是计算边缘分布$P(\pmb{x}_F)$或条件分布$P(\pmb{x}_F|\pmb{x}_E)$。由条件概率可知，

$$
P(\pmb{x}_F|\pmb{x}_E)=\frac{P(\pmb{x}_F,\pmb{x}_E)}{P(\pmb{x}_E)}=\frac{P(\pmb{x}_F,\pmb{x}_E)}{\sum_{\pmb{x}_F}P(\pmb{x}_F,\pmb{x}_E)}
$$

上式中，联合分布可以由图模型得到，所以推断问题的关键就是如何计算分母中的边缘分布，也就是

$$
P(\pmb{x}_E)=\sum_{\pmb{x}_F}P(\pmb{x}_F,\pmb{x}_E)
$$(edge-distribution)
### 精确推断

#### 变量消除法

&emsp;&emsp;精确推断利用图模型所描述的条件独立性来减少计算目标概率所需的计算量。变量消除法是最直观的精确推断算法也是其它精确推断算法的基础。下面通过一个例子来展示该方法。


:::{admonition} **例**3. 下图给出了一个有向图模型
:class: dropdown

```{mermaid}
---
caption: Fig 4. 贝叶斯网络结构示例 
align: center
---
flowchart LR
  x1(("x1")) --> x2(("x2"))
  x1 -. m12 .-> x2
  x2 --> x3(("x3"))
  x2 -.m23.-> x3
  x3 --> x4(("x4"))  
  x3 --> x5(("x5"))
  x3 -.m35.-> x5
  x4 -.m43.-> x3
 

```
假设要推断目标是计算边缘分布$P(x_5)$，很显然只需要消去变量$\{x_1,...,x_4\}$，即

$$
\begin{split}
P(x_5)&=\sum_{x_4}\sum_{x_3}\sum_{x_2}\sum_{x_1}P(x_1,x_2,x_3,x_4,x_5)\\
&=\sum_{x_4}\sum_{x_3}\sum_{x_2}\sum_{x_1}P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3)P(x_5|x_3)
\end{split}
$$

如果采用$\{x_1,x_2,x_4,x_3\}$的顺序计算，则有，

$$
\begin{split}
P(x_5)&=\sum_{x_4}\sum_{x_3}\sum_{x_2}\sum_{x_1}P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3)P(x_5|x_3)\\
&=\sum_{x_3}P(x_5|x_3)\sum_{x_4}P(x_4|x_3)\sum_{x_2}P(x_3|x_2)\sum_{x_1}P(x_2|x_1)P(x_1)
\end{split}
$$

引入记号$m_{ij}(x_j)$表示求各过程的中间结果，下标$i$表示对$x_i$求和，$j$表示该项剩余的其它变量。显然$m_{ij}(x_j)$是$x_j$的函数。对上式不断执行此过程，可得，

$$
\begin{split}
P(x_5)&=\sum_{x_3}P(x_5|x_3)\sum_{x_4}P(x_4|x_3)\sum_{x_2}P(x_3|x_2)\underbrace{\sum_{x_1}P(x_2|x_1)P(x_1)}_{m_{12}(x_2)}\\
&=\sum_{x_3}P(x_5|x_3)\sum_{x_4}P(x_4|x_3)\sum_{x_2}P(x_3|x_2)m_{12}(x_2)\\
&=\sum_{x_3}P(x_5|x_3)\sum_{x_4}P(x_4|x_3)m_{23}(x_3)\\
&=\sum_{x_3}P(x_5|x_3)m_{23}(x_3)m_{43}(x_3)\\
&=m_{35}(x_5)
\end{split}
$$

显然，最后的计算结果是关于$x_5$的函数。事实上，该方法对无向图模型仍然适用。

:::

&emsp;&emsp;变量消去法通利用乘法对加法的分配律，把多个变量的积的求各问题转化为部分变量交替进行求积和求各的问题。这种转化使得每次的求和与求积运算限制在局部，仅与部分变量有关，从而提高了计算效率。


  :缺陷: 如果需要计算多个边缘分布，重复使用变量消除法会造成大量的冗余计算。


#### 信念传播

&emsp;&emsp;信念传播将变量消除法中的求和操作当作一个消息传递过程，可以解决求解多个边缘分布重复计算的问题。变量消除法的求和操作可以视为以下过程，

$$
m_{ij}(x_j)=\sum_{x_i}\psi(x_i,x_j)\prod_{k\in\{ n(i)\backslash j\}}m_{ki}(x_i)
$$(sum-operator)

该操作在信念传播算法中被当作为结点$x_i$向结点$x_j$传递了一个消息$m_{ij}(x_j)$。图4的虚线描述了$P(x_5)$计算的消息传递过程。可以看出，消息传递操作仅与变量$x_i$以及邻接结点直接相关，也就是计算限制在图的局部进行。

&emsp;&emsp;一个结点仅在接收到来自其它所有邻接结点的消息后才能向另一接点发送消息，且结点的边缘分布正比于它所接收到的消息的乘积，即

$$
P(x_i)\propto \prod_{k\in n(i)}m_{ki}(x_i)
$$(edge-dist-sum-prod)

例如，图4的结点$x_3$要向$x_5$发送消息，必须先接收到来自结点$x_2$和$x_4$的消息，且传递给$x_5$的消息$m_{35}(x_5)$正好为概率$P(x_5)$。

&emsp;&emsp;如果图中没有环，则信念传播算法经过**两个步骤**即可完成所有的消息传递，从而能计算所有变量的边缘分布：
1. 指定一个根结点，从所有叶结点开始向根结点传递消息，直到根结点收到所有邻接结点的消息； 
2. 从根结点开始向叶结点传递消息，直到所有叶结点收到消息


```{mermaid}
---
caption: Fig 5. 信念传播算法示例，选$x_1$为根结点：实线为第1步消息传递；虚线为第2步。  
align: center
---
flowchart LR
  x1(("x1")) -.m12.-> x2(("x2"))
  x2 -- m21 --> x1
  x2 -.m23.-> x3(("x3"))
  x3 --m32--> x2
  x3 -.m34.-> x4(("x4"))  
  x3 -.m35.-> x5(("x5"))
  x5 --m53--> x3
  x4 --m43--> x3
 

```
经过上图的两步消息传递后，图的每个结点都有着不同方向的两条消息，基于这些消息和等式{eq}`edge-dist-sum-prod`即可计算得到所有变量的边缘分布。


### 近似推断

&emsp;&emsp;近似推断主要采用采样（sampling）或变分推断（variational inference）来实现。精确推断方法通常需要很大的计算开销，实际应用中近似推断更为常见。

#### MCMC采样

&emsp;&emsp;通过概率图模型可以得到一些感兴趣变量的概率分布，在很多实际应用中，对分布本身并没有多大兴趣，而是希望通过这些概率分布计算某些期望，并根据这些期望做出决策。例如图4的贝叶斯网，进行推断的目标可能是为了计算变量$x_5$的期望。如果直接计算或近似这个期望比推断概率分布更容易，则直接计算将使推断问题求解更高效。对于概率图模型来说，问题的关键就在于如何高效的基于图模型描述的概率分布来获取样本。

&emsp;&emsp;概率图模型最常用的采样技术是马尔可夫链蒙特卡罗方法（Markov Chain Monte Carlo, MCMC）。假设有随机变量$x\sim p(x)$，函数$f:X\rightarrow \mathbb{R}$，则$f(x)$的期望为，

$$
p(f)=\mathbb{E}_p[f(x)]=\int_x f(x)p(x)dx
$$(efx)

若$p(x)$易采样且$f(x)$不易积分过于复杂，则可以通过Monte Carlor积分方法近似的计算期望如下，

$$
\hat{p}(f)\approx \frac1N\sum_{i=1}^N f(x_i), \quad x_i \sim p(x)
$$(mc-integrate)

&emsp;&emsp;但实际应用中，图模型构造的密度函数$p(\pmb{x})$过于复杂，从而导致采样独立同分布的样本非常困难。幸运的是，随机过程理论有一个**结论**：

{attribution="非周期马尔可夫链"}
> 马尔可夫链依条件（细致平稳条件）收敛于平稳分布$q(\pmb{x})$，收敛后得到的跳转序列$\{\pmb{x}_n,\pmb{x}_{n+1},...\}$恰好是$q(\pmb{x})$的样本序列。

那么一个绝秒的想法是：构造一个马尔可夫链，使得它的平稳分布恰好为$p(\pmb{x})$。而这就是MCMC方法的**基本思想**。

&emsp;&emsp;MCMC方法如何构造马尔可夫链将会产生不同的MCMC算法。一般来说，使用MCMC方法构造一条马尔可夫链产生符合图模型的概率分布的样本集，再利用这些样本对所需要的函数进行估计。常用的MCMC算法有，**Metropolis-Hastings算法**、**Gibbs算法**等[^mcmc]。


#### 变分推断

&emsp;&emsp;变分推断使用已知简单分布来逼近需要推断的复杂分布，通过限制近似分布得到局部最优确定解。


##### 盘式记法

```{mermaid}
---
caption: Fig 4. 普通变量关系图 
align: center
---
%%{init: { "theme": "default" } }%%
graph TD   
  z(("z")) --> x1(("x1"))
  z --> x2(("x2"))
  z --> x3(("..."))
  z --> x4(("xn"))
  subgraph 观测变量
  x1
  x2
  x3
  x4
  end


```

&emsp;&emsp;概率图模型的盘式记法将相互独立、由相同机制产生的多个变量放在一个方框内，并在方框内标注重复出现的次数$N$。在很多学习任务中，对属性变量使用盘式记法将使得图表示非常简洁。例如：




```{mermaid}
---
caption: Fig 5. 盘式记法 
align: center
---
graph LR   
  z(("z")) --> x1(("x"))
  
  subgraph  N
  x1
  
  end


```
上图观测变量的联合分布概率密度函数是,

$$
p(\pmb{x}|\Theta)=\prod_{i=1}^N\sum_{\pmb{z}}p(x_i,\pmb{z}|\Theta),\quad \ln p(\pmb{x}|\Theta)=\sum_{i=1}^N\ln\left\{\sum_{\pmb{z}}p(x_i,\pmb{z}|\Theta)\right\}
$$

其中，$\pmb{x}=\{x_1,...,x_n\}$，$\Theta$是$\pmb{x},\pmb{z}$服从的分布的参数。

&emsp;&emsp;一般来说，推断和学习的任务主要是由观察到的变量$\pmb{x}$来估计隐变量$\pmb{z}$和分布参数$\Theta$，即求解$p(\pmb{z}|\pmb{x},\Theta)$和$\Theta$。概率模型的参数估计通常通过最大化对数似然函数求解，对于具有隐变量的概率模型可以使用EM算法求解。

- 在E步，根据$t$时刻的参数$\Theta^t$对$p(\pmb{x}|\Theta^t)$进行推断，并计算联合似然函数$p(\pmb{x},\pmb{z}|\Theta)$; 
- 在M步，最优化$Q$函数的参数$\Theta$。

$$
\begin{split}
Q^{t+1}&=\arg\max\limits_{\Theta}Q(\Theta;\Theta^t)\\
&=\arg\max\limits_{\Theta}\sum_{\pmb{z}}p(\pmb{z}|\pmb{x},\Theta^t)\ln p(\pmb{x},\pmb{z}|\Theta)\\
&=\arg\max\limits_{\Theta} \mathbb{E}_\pmb{z}[\ln p(\pmb{x},\pmb{z}|\Theta)]
\end{split}
$$(Q-function)

&emsp;&emsp;$Q$函数实际上是完全数据对数似然$\ln p(\pmb{x},\pmb{z}|\Theta)$在分布$p(\pmb{z}|\pmb{x},\Theta^t)$下的期望。当分布与变量$\pmb{z}$的后验分布相等时，$Q$函数近似等于完全数据对数似然。最终，EM算法可以获得稳定优化的参数$\Theta$，隐变量$\pmb{z}$的分布也可能通过该参数得到。

&emsp;&emsp;事实上$p(\pmb{z}|\pmb{x},\Theta^t)$是隐变量$\pmb{z}$的一个近似分布，如果将这个近似分布用$q(\pmb{z})$表示，则可以得出以下结论，

$$
\ln p(\pmb{x})=\mathcal{L}(q)-\mathrm{KL}(q||p)
$$(low-bound)

其中，

$$
\mathcal{L}(q)=\int q(\pmb{z})\ln \frac{p(\pmb{x},\pmb{z})}{q(\pmb{z})}d\pmb{z},\quad \mathrm{KL}(q||p)=\int q(\pmb{z})\ln\frac{p(\pmb{z}|\pmb{x})}{q(\pmb{z})}
$$

注意:

$$
\int q(\pmb{z})\ln p(\pmb{x})=\ln p(\pmb{x})
$$

只有KL散度为0时，$\mathcal{L}(q)$才接近对数似然。因为KL散度非负，因此$\mathcal{L}(q)$也称为对数似然的一个下界。

##### 均值场方法

&emsp;&emsp;现实中，E步对$p(\pmb{x}|\Theta^t)$的推断很可能难以实现，此时可借助变分推断。通常假设$\pmb{z}$服从分布，

$$
q(\pmb{z})=\prod_{i=1}^M q_i(\pmb{z}_i)
$$(q-factor)

可以将多变量$\pmb{z}$拆解为一系列相互独立的多变量$\pmb{z}_i$，简写$q_i(\pmb{z}_i)$为$q_i$代入可得,

$$
\begin{split}
\mathcal{L}(q)&=\int_{\pmb{z}}\prod_i q_i\left\{\ln p(\pmb{x},\pmb{z})-\sum_i\ln q_i \right\}d\pmb{z}\\
&=\int_{\pmb{z}_{-j}}\int_{\pmb{z}_j} q_j\prod_{i\neq j}q_i\left\{\ln p(\pmb{x},\pmb{z})-\sum_i\ln q_i \right\}d\pmb{z}\\
&=\int_{\pmb{z}_j} q_j\int_{\pmb{z}_{-j}}\prod_{i\neq j}q_i\left\{\ln p(\pmb{x},\pmb{z})-\sum_i\ln q_i \right\}d\pmb{z}\\
&=\int_{\pmb{z}_j} q_j \underbrace{\int_{\pmb{z}_{-j}}\prod_{i\neq j}q_i\left\{\ln p(\pmb{x},\pmb{z})\right\}}_{\ln\tilde{p}(\pmb{x},\pmb{z})\triangleq\mathbb{E}_{i\neq j}[\ln p(\pmb{x},\pmb{z})]}d\pmb{z} - \int_{\pmb{z}_j} q_j\int_{\pmb{z}_{-j}}\prod_{i\neq j}q_i\left\{\sum_i\ln q_i\right\}d\pmb{z}\\
&=\int q_j \mathbb{E}_{i\neq j}[\ln p(\pmb{x},\pmb{z})]d\pmb{z}_j-\int q_j \ln q_j d\pmb{z}_j+\mathrm{const}
\end{split}
$$(mean-field)

&emsp;&emsp;我们关心的是分布$q_j$，因此可以固定$q_{i\neq j}$再对$\mathcal{L}$最大化。可以发现等式{eq}`mean-field`等价于$-\mathrm{KL}(q_j||\tilde{p}(\pmb{x},\pmb{z}))$，即当$q_j=\tilde{p}(\pmb{x},\pmb{z})$时$\mathcal{L}$最大。因此可知变量子集$\pmb{z}_j$服从的最优分布$q_j^*$应满足，

$$
\ln q_j^*(\pmb{z}_j)=\mathbb{E}_{i\neq j}[\ln p(\pmb{x},\pmb{z})]+\mathrm{const}
$$

即，

$$
q_j^*(\pmb{z}_j)=\frac{\exp(\mathbb{E}_{i\neq j}[\ln p(\pmb{x},\pmb{z})])}{\int \exp(\mathbb{E}_{i\neq j}[\ln p(\pmb{x},\pmb{z})]) d\pmb{z}_j}
$$(optimize-solution)

&emsp;&emsp;最终，在满足条件{eq}`q-factor`下，变量子集$\pmb{z}_j$最接近的真实分布由{eq}`optimize-solution`得到。

&emsp;&emsp;通过恰当的分割独立变量子集$\pmb{z}_j$并选择$q_j$服从的分布，$\mathbb{E}_{i\neq j}[\ln p(\pmb{x},\pmb{z})]$往往有闭式解，这使得基于{eq}`optimize-solution`能高效对隐变量$\pmb{z}$进行推断。事实上，对变量$\pmb{z}_j$分布$q_j^*$进行估计时整合了$\pmb{z}_j$之外的其他$\pmb{z}_{i\neq j}$的信息，这是通过联合似然函数$\ln p(\pmb{x},\pmb{z})$在$\pmb{z}_j$之外的隐变量分布上求期望得到的，因此也被称为**平均场(mean field)**方法。


[^mcmc]: [MCMC采样](https://sspuiip.github.io/ml/mathmodel/mcmc.html#mcmc)




