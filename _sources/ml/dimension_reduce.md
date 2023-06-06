# 表示学习

&emsp;&emsp;一般来说，机器学习中的原始数据是高维数据。高维数据往往具有复杂性、冗余性等特点。高维空间会有很多不一样的特征，也称之为维度灾难。

&emsp;&emsp;- 大多数数据对象之间相距都很远(高维数据有很大可能非常稀疏)

&emsp;&emsp;- 理论上，通过增大训练集，使训练集达到足够密度，是可以避免维度灾难的。但实践中，要达到给定密度，所需的训练数据随着维度的增加呈指数式上升

&emsp;&emsp;为了避免维度灾难，以及找到问题求解最合适的数据表示形式，需要研究原有数据的表示问题，这一过程也称之为**表示学习**。


## 主成分分析

&emsp;&emsp;主成分分析(Principal Component Analysis, PCA)是一种通过某种正交变换将一组可能存在相关关系的变量转换为一组线性不相关的变量。对于训练数据，

$$
\pmb{X}=\begin{pmatrix}|&|&\cdots &| \\
\pmb{x}_1 & \pmb{x}_2 &  \cdots & \pmb{x}_m\\ |&|&\cdots &|\end{pmatrix}
$$

其中，$\pmb{x}_i=(x_{i1},...,x_{in})^\top$。PCA的目标是找到一个变换矩阵$\pmb{W}_{n\times d}$使得,

$$
\min\limits_{\pmb{W}}\quad \lVert\pmb{X}_{new}-\pmb{X}\pmb{W}\rVert
$$






