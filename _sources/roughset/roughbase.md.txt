# 粗糙集基础

## Pawlak粗糙集
&emsp;&emsp;Pawlak粗糙集主要围绕等价关系展开。给定一个数据集$D=\{(\pmb{x}_1,y_1),...,(\pmb{x}_m,y_m)\}$，其中$\pmb{x}_i$为categorical变量，即所有变量值都是名义型的；$y_i$为类别，也是categorical变量。粗糙集首先根据特征矩阵$\pmb{X}=(\pmb{x}_1^\top,...,\pmb{x}_m^\top)^\top$，产生所有样本的等价类集合，即，

$$
[\pmb{x}]_R = \{\pmb{x}'| \forall \pmb{x}'\in U,\quad\pmb{x}R\pmb{x}' \}
$$(equivalence-cluster)

其中，$U$是$\pmb{x}$的全体。

&emsp;&emsp;假设所有样本根据等价关系$R$拆成分$k=3$个等价类簇，例如，$\{\{\pmb{x}_1,\pmb{x}_5,\pmb{x}_2\},\{\pmb{x}_3,\pmb{x}_4\},\{\pmb{x}_6,\pmb{x}_7,\pmb{x}_8\}\}$，则数据集$D$可以表达为等价类簇的集合：即

$$
D = \{[\pmb{x}_i]_R\}_{i=1}^m = \{\{\pmb{x}_1,\pmb{x}_5,\pmb{x}_2\},\{\pmb{x}_3,\pmb{x}_4\},\{\pmb{x}_6,\pmb{x}_7,\pmb{x}_8\}\}
$$

### 上下近似

&emsp;&emsp;若任意给定一个集合$A\subseteq U$，则根据等价关系$R$划分后的类簇关系，可以对集合$A$做以下操作，

$$
\begin{split}
\underline{\textrm{apr}}_R(A)&=\{ \pmb{x}|\quad \forall\pmb{x}\in U,[\pmb{x}]_R \cap A \neq \emptyset   \}\\
\overline{\textrm{apr}}_R(A)&=\{ \pmb{x}|\quad \forall\pmb{x}\in U,[\pmb{x}]_R \subseteq A   \}\\
\end{split}
$$(low-appr-op)

分别称为**上、下近似算子**。同时，称$A$是全体样本$U$在等价关系$R$划分下的**正域**、**负域**，即

$$
\begin{split}
\textrm{POS}(A)&\triangleq \underline{\textrm{apr}}_R(A)\\
\textrm{NEG}(A)&\triangleq U-\overline{\textrm{apr}}_R(A)
\end{split}
$$(lu-operator)

以及**边界域**，

$$
\textrm{BND}(A)\triangleq \overline{\textrm{apr}}_R(A)-\underline{\textrm{apr}}_R(A)
$$(bnd-operator)

&emsp;&emsp;若边界域不为空，则称集合$A$为**粗糙集**。

## 概率粗糙集

&emsp;&emsp;在Pawlak粗糙集的基础上，引入一个概率测度$P$，

$$
P(A|[\pmb{x}])\triangleq \frac{[\pmb{x}]\cap A}{|[\pmb{x}]|}
$$(prob-measure)

则可以对{eq}`low-appr-op`进行重定义，即

$$
\begin{split}
\underline{\textrm{apr}}_{(\alpha,\beta)}(A)&\triangleq \{ \pmb{x}|\forall \pmb{x}\in U,P(A|[\pmb{x}])\geq \alpha \}\\
\overline{\textrm{apr}}_{(\alpha,\beta)}(A)&=\{\pmb{x}|\forall \pmb{x}\in U,P(A|[\pmb{x}])>\beta \}
\end{split}
$$(prob-roughset)

其中，$0\le\beta<\alpha\le 1$。则有，

$$
\begin{split}
\textrm{POS}_{(\alpha,\beta)}(A)&\triangleq \underline{\textrm{apr}}_{(\alpha,\beta)}(A)\\
\textrm{NEG}_{(\alpha,\beta)}(A)&\triangleq U-\overline{\textrm{apr}}_{(\alpha,\beta)}(A)=\{\pmb{x}|\forall \pmb{x}\in U,P(A|[\pmb{x}])>\beta \}
\end{split}
$$(prob-lu-operator)

以及，

$$
\textrm{BND}_{(\alpha,\beta)}(A)=\{\beta<P(A|[\pmb{x}])<\alpha\}
$$(prob-bnd-operator)

&emsp;&emsp;若边界域不为空，则称集合$A$为**概率粗糙集**。

## 决策粗糙集

&emsp;&emsp;若对决策代价进行建模，即

:::{table} 决策损失函数（二类问题为例）
:width: 750px
:align: center
:widths: grid
| 样本$x$与集合$X$的关系决策    | 接受($a_p$)    | 延迟($a_b$)    | 拒绝($a_n$)|
| :---: | :---: | :---: | :---: |
| 决策1：$x\in X$| $\lambda_{pp}$| $\lambda_{bp}$ | $\lambda_{np}$ |
| 决策2：$x \in \neg X$|$\lambda_{pn}$| $\lambda_{bn}$ | $\lambda_{nn}$ |

:::

其中，$\lambda_{pp},\lambda_{bp},\lambda_{np}$表示决策1，即判定$x\in X$时采取行动$a_p, a_b,a_n$时的收益。与之类似$\lambda_{pn},\lambda_{bn},\lambda_{nn}$为决策2时采取3个行动所带来的收益。同时，假设以上收益满足以下两个条件，

- $0\le \lambda_{pp}\le\lambda_{bp}<\lambda_{np}$， $0\le \lambda_{nn}\le\lambda_{bn}<\lambda_{pn}$
- $\frac{\lambda_{np}-\lambda_{bp}}{\lambda_{bn}-\lambda_{nn}} >\frac{\lambda_{bp}-\lambda_{pp}}{\lambda_{pn}-\lambda_{bn}}$

&emsp;&emsp;则有，对于概率粗糙集的三条决策规则，阈值$\alpha,\beta$的最优取值为，

$$
\begin{split}
\hat{\alpha}&=\frac{\lambda_{pn}-\lambda_{bn}}{(\lambda_{pn}-\lambda_{bn})+(\lambda_{bp}-\lambda_{pp})}\\
\hat{\beta}&=\frac{\lambda_{bn}-\lambda_{nn}}{(\lambda_{bn}-\lambda_{nn})+(\lambda_{np}-\lambda_{bp})}\\
\end{split}
$$(deci-prob-roughset)

&emsp;&emsp;最终，根据{eq}`deci-prob-roughset`所计算的阈值，得到概率粗糙集的三条决策规则：

1. 如果$P(X|[\pmb{x}])\ge\hat{\alpha}$，则$x\in \textrm{POS}(X)$；
2. 如果$\hat{\beta}< P(X|[\pmb{x}])<\hat{\alpha}$，则$x\in \textrm{BND}(X)$；
3. 如果$P(X|[\pmb{x}])\le\hat{\beta}$，则$x\in \textrm{NEG}(X)$；

该粗糙集模型即为**决策粗糙集**。