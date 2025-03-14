# 决策树

&emsp;&emsp;**决策树**是一种模拟人决策过程的树形结构。例如：如果有以下一段对话，
>   女儿：多大年纪了？<br>
  母亲：26。<br>
  女儿：长的帅不帅？<br>
  母亲：挺帅的。<br>
  女儿：收入高不？<br>
  母亲：不算很高，中等情况。<br>
  女儿：是公务员不？<br>
  母亲：是，在税务局上班呢。<br>
  女儿：那好，我去见见。<br>

上述对话体现了人类做出见/不见决策的一个过程。根据对话的顺序，该决策过程可以用下图来刻画。

```{mermaid}
---
caption: Fig 1. 决策过程模拟树。  
align: center
---
%%{
    init: {
        'theme':'base',
        'themeVariables': {
            'fontSize': 8px
        }
    }
}%%
flowchart LR
  id1((年龄)) -- <=30--> id2((长相))
  id1 -- >30 --> id3[不见]
  id2 -- 帅 --> id4((收入))
  id2 -- 不帅 --> id5[不见]
  id4 -- 高 --> id6[见]
  id4 -- 中 --> id7((公务员))
  id4 -- 低 --> id8[不见]
  id7 -- 是 --> id9[见]
  id7 -- 不是 --> id10[不见]
 
```

&emsp;&emsp;**决策树在构建过程中有个基本问题需要解答**。如上图所示，为什么选择年龄做为第1个分裂的结点而不是其它特征？也就是如何确定分裂特征的顺序？这一问题也称为**特征选择**问题。

&emsp;&emsp;依据特征选择策略的不同，决策树算法大至有ID3,C4.5,CART等不同的决策树构建算法。这些算法的基本流程框架如下所示：

|决策树学习算法基本框架|
|:---|
|**输入**: 训练集$D=\{(\pmb{x}_1,y_1),(\pmb{x}_2,y_2),...,(\pmb{x}_m,y_m)\}$，属性集$A=\{a_1,a_2,...,a_d\}$ |
|**输出**: 以*root*为根的一棵决策树|
|**过程**: *decisionTree(D,A)*|
|1: 生成结点*root* <br>2: **if**($D$中样本属于同一类别$C$){ <br>3: &emsp;&emsp; return; <br>4: }<br>5: **if**($A=\emptyset$ or $D$中样本在属性$A$取值一致){<br>6: &emsp;&emsp;将root标记为叶结点，类别为$D$中样本数最多的类; return;<br>7: }<br>8: 根据属性挑选规则，在属性集$A$中挑选最优属性$a$;<br>9: **for**($a$的每一个属性值$a^v$){<br>10:&emsp;&emsp;为root生成一个分支结点，该结点的样本集$D_v$为$a==a^v$的所有样本;<br>11:&emsp;&emsp;**if**($D_v$为空){<br>12:&emsp;&emsp;&emsp;&emsp;将分支结点(子结点)标记为叶结点;类别为$D$中样本最多的类别; return;<br>13:&emsp;&emsp;**else**<br>14:&emsp;&emsp;&emsp;&emsp;以dicisionTree($D_v$,A-$\{a\})$为分支结点继续分裂;<br>15:&emsp;&emsp;}<br>16:}|


## ID3

&emsp;&emsp;ID3（Iterative Dichotomiser 3）算法是一种经典的决策树学习算法，由Ross Quinlan于1986年提出。该算法的主要目的是通过构建一个决策树模型来对样本数据进行分类。ID3算法的核心思想是基于**信息增益**（Information Gain）来选择最佳的属性作为决策树的节点，以此来实现对数据的划分。

&emsp;&emsp;**定义（信息熵）**. 假设有离散随机变量（连续型同样适用，只是公式的表示形式不一样）$X$其取值概率分别为$\{(x_1,p_1),...,(x_n,p_n)\}$(s.t.$\sum_i p_i=1$)，则该随机变量的信息熵(不确定性)为，

$$
\textrm{H}(X)=-\sum_i^n p_i\log p_i
$$(entropy-def)

&emsp;&emsp;**定义（条件熵）**. 假设有离散随机变量$X,Y$，则条件熵为(记$p(y_i)\triangleq P(Y=y_i)$)，

$$
      \begin{split}
        \textrm{H}(X|Y)&=\sum_{y_i\in Y}p(y_i)\textrm{H}(X|y_i)\\
        &=-\sum_{y_i\in Y}p(y_i)\sum_{x_j\in X}P(x_j|y_i)\log P(x_j|y_i)
      \end{split}
$$(cond-entropy)


&emsp;&emsp;**定义（信息增益）**. 随机变量$X$的信息熵与条件熵$X|Y$的差异即为信息增益，即

$$
        \textrm{IG}(X,Y)=\textrm{H}(X)-\textrm{H}(X|Y)
$$(information-gain)

### ID3构建算法

&emsp;&emsp;ID3构建算法只需要对上一节的决策树学习算法的第8行进行修改，即为ID3学习算法。

|ID3决策树学习算法基本框架|
|:---|
|**输入**: 训练集$D=\{(\pmb{x}_1,y_1),(\pmb{x}_2,y_2),...,(\pmb{x}_m,y_m)\}$，属性集$A=\{a_1,a_2,...,a_d\}$ |
|**输出**: 以*root*为根的一棵决策树|
|**过程**: *decisionTree(D,A)*|
|1: 生成结点*root* <br>2: **if**($D$中样本属于同一类别$C$){ <br>3: &emsp;&emsp; return; <br>4: }<br>5: **if**($A=\emptyset$ or $D$中样本在属性$A$取值一致){<br>6: &emsp;&emsp;将root标记为叶结点，类别为$D$中样本数最多的类; return;<br>7: }<br>8: **计算所有属性的信息增益，在属性集$A$中挑选最大增益属性**$a$;<br>9: **for**($a$的每一个属性值$a^v$){<br>10:&emsp;&emsp;为root生成一个分支结点，该结点的样本集$D_v$为$a==a^v$的所有样本;<br>11:&emsp;&emsp;**if**($D_v$为空){<br>12:&emsp;&emsp;&emsp;&emsp;将分支结点(子结点)标记为叶结点;类别为$D$中样本最多的类别; return;<br>13:&emsp;&emsp;**else**<br>14:&emsp;&emsp;&emsp;&emsp;以dicisionTree($D_v$,A-$\{a\})$为分支结点继续分裂;<br>15:&emsp;&emsp;}<br>16:}|

### ID3应用案例

&emsp;&emsp;应用ID3算法构建一棵决策树，该案例使用UCI数据集weather，内容如下表所示：

| **outlook** | **temperature** | **humidity** | **windy** | **play** |
|:-----------:|:---------------:|:------------:|:---------:|:--------:|
| sunny       | hot             | high         | FALSE     | no       |
| sunny       | hot             | high         | TRUE      | no       |
| overcast    | hot             | high         | FALSE     | yes      |
| rainy       | mild            | high         | FALSE     | yes      |
| rainy       | cool            | normal       | FALSE     | yes      |
| rainy       | cool            | normal       | TRUE      | no       |
| overcast    | cool            | normal       | TRUE      | yes      |
| sunny       | mild            | high         | FALSE     | no       |
| sunny       | cool            | normal       | FALSE     | yes      |
| rainy       | mild            | normal       | FALSE     | yes      |
| sunny       | mild            | normal       | TRUE      | yes      |
| overcast    | mild            | high         | TRUE      | yes      |
| overcast    | hot             | normal       | FALSE     | yes      |
| rainy       | mild            | high         | TRUE      | no       |


:::{admonition} **计算过程**
:class: dropdown


<br>

**第1步. 根据信息增益选择分裂属性，对数据集D进行分裂**。

&emsp;&emsp;**1.1 计算分类属性`play`的信息熵**。

$$
\textrm{H(play)}=-\frac{5}{14}\log\frac{5}{14}-\frac{9}{14}\log\frac{9}{14}=0.94
$$

&emsp;&emsp;**1.2 计算分类属性对所有其它属性的条件熵**。

$$
\begin{split}
\textrm{H(play|outlook)}&=-\frac{5}{14}\left(\frac25\log\frac25+\frac35\log\frac35\right)-\frac{4}{14}\left(1\log 1+0\log 0\right)-\frac{5}{14}\left(\frac25\log\frac25+\frac35\log\frac35\right)=0.69\\
\textrm{H(play|temperatur)}&=-\frac{4}{14}\left(\frac12\log\frac12+\frac12\log\frac12\right)-\frac{6}{14}\left(\frac46\log\frac46 +\frac26\log\frac26\right)-\frac{4}{14}\left(\frac13\log\frac13+\frac23\log\frac23\right)=0.91\\
\textrm{H(play|humidity)}&=-\frac{7}{14}\left(\frac37\log\frac37+\frac47\log\frac47\right)-\frac{7}{14}\left(\frac17\log\frac17 +\frac67\log\frac67\right)=0.78\\
\textrm{H(play|windy)}&=-\frac{6}{14}\left(\frac36\log\frac36+\frac36\log\frac36\right)-\frac{8}{14}\left(\frac68\log\frac68 +\frac28\log\frac28\right)=0.89\\
\end{split}
$$


&emsp;&emsp;**1.3 计算所有属性的信息增益**。

$$
\begin{split}
\textrm{IG(play,outlook)}&=\textrm{H(play)}-\textrm{H(play|outlook)}=0.24\\
\textrm{IG(play,temperatur)}&=\textrm{H(play)}-\textrm{H(play|temperatur)}=0.02\\
\textrm{IG(play,humidity)}&=\textrm{H(play)}-\textrm{H(play|humidity)}=0.15\\
\textrm{IG(play,windy)}&=\textrm{H(play)}-\textrm{H(play|windy)}=0.04\\
\end{split}
$$

&emsp;&emsp;**1.4 选择信息增益最大的属性`outlook`分裂**。

&emsp;&emsp;当`outlook=overcast`时，该子集的决策属性值一致，都为`yes`，因此不必对该分支继续分裂；相反地，D1·D2还需进一步分裂。分裂结果如下图所示：

```{mermaid}
---
caption: Fig 2. 根据属性*outlook*分裂数据集。  
align: center
---
%%{
    init: {
        'theme':'base',
        'themeVariables': {
            'fontSize': 8px
        }
    }
}%%
flowchart LR
  id1((outlook)) -- overcast --> id2[YES]
  id1 -- rainy --> id3[(D1)]
  id1 -- sunny --> id4[(D2)]
 
```

**第2步. 根据信息增益选择分裂属性，对数据集D1进行分裂**。

&emsp;&emsp;**2.1 计算分类属性`play`的信息熵**。第1步已计算，直接使用上一步的结果。

&emsp;&emsp;**2.2 计算分类属性对所有未选择的属性的条件熵**。

$$
\begin{split}
\textrm{H(play|temperatur)}&=-\frac{2}{5}\left(\frac12\log\frac12+\frac12\log\frac12\right)-\frac{3}{5}\left(\frac13\log\frac13 +\frac23\log\frac23\right)=0.951\\
\textrm{H(play|humidity)}&=-\frac{2}{5}\left(\frac12\log\frac12+\frac47\log\frac47\right)-\frac{3}{5}\left(\frac13\log\frac13 +\frac23\log\frac23\right)=0.951\\
\textrm{H(play|windy)}&=-\frac{2}{5}\left(1\log 1+0 \log 0\right)-\frac{3}{5}\left(1\log 1+0 \log 0\right)=0.00\\
\end{split}
$$


&emsp;&emsp;**2.3 计算所有属性的信息增益**。

$$
\begin{split}
\textrm{IG(play,temperatur)}&=\textrm{H(play)}-\textrm{H(play|temperatur)}=0.02\\
\textrm{IG(play,humidity)}&=\textrm{H(play)}-\textrm{H(play|humidity)}=0.02\\
\textrm{IG(play,windy)}&=\textrm{H(play)}-\textrm{H(play|windy)}=0.97\\
\end{split}
$$

&emsp;&emsp;**2.4 选择信息增益最大的属性`windy`分裂**。

&emsp;&emsp;当`windy=yes`时，该子集的决策属性值一致，都为`no`；当`windy=no`时，该子集的决策属性值一致，都为`yes`，因此不必对所有分支继续分裂。至此，D1分裂结束。分裂结果如下图所示：

```{mermaid}
---
caption: Fig 3. 根据属性*windy*分裂数据集D1。  
align: center
---
%%{
    init: {
        'theme':'base',
        'themeVariables': {
            'fontSize': 8px
        }
    }
}%%
flowchart LR
  id1((outlook)) -- overcast --> id2[YES]
  id1 -- rainy --> id3[(windy)]
  id1 -- sunny --> id4[(D2)]
  id3 -- yes --> id5[NO]
  id3 -- no --> id6[YES]
 
```

**第3步. 根据信息增益选择分裂属性，对数据集D2进行分裂**。

&emsp;&emsp;**3.1 计算分类属性`play`的信息熵**。第1步已计算，直接使用上一步的结果。

&emsp;&emsp;**3.2 计算分类属性对所有未选择的属性的条件熵**。

$$
\begin{split}
\textrm{H(play|temperatur)}&=-\frac{2}{5}\left(1\log 1+0 \log 0\right)-\frac{2}{5}\left(\frac12\log\frac12 +\frac12\log\frac12\right)-\frac{1}{5}\left(1\log 1+0 \log 0 \right)=0.4\\
\textrm{H(play|humidity)}&=-\frac{2}{5}\left(1\log 1+0 \log 0\right)-\frac{3}{5}\left(1\log 1+0 \log 0\right)=0.0\\
\textrm{H(play|windy)}&=-\frac{2}{5}\left(\frac12\log\frac12 +\frac12\log\frac12\right)-\frac{3}{5}\left(\frac13\log\frac13 +\frac23\log\frac23\right)=0.95\\
\end{split}
$$


&emsp;&emsp;**3.3 计算所有属性的信息增益**。

$$
\begin{split}
\textrm{IG(play,temperatur)}&=\textrm{H(play)}-\textrm{H(play|temperatur)}=0.57\\
\textrm{IG(play,humidity)}&=\textrm{H(play)}-\textrm{H(play|humidity)}=0.97\\
\textrm{IG(play,windy)}&=\textrm{H(play)}-\textrm{H(play|windy)}=0.02\\
\end{split}
$$

&emsp;&emsp;**3.4 选择信息增益最大的属性`humidity`分裂**。

&emsp;&emsp;当`humidity=high`时，该子集的决策属性值一致，都为`no`；当`humidity=normal`时，该子集的决策属性值一致，都为`yes`，因此不必对所有分支继续分裂。至此，D2分裂结束。分裂结果如下图所示：

```{mermaid}
---
caption: Fig 4. 根据属性*humidity*分裂数据集D2。  
align: center
---
%%{
    init: {
        'theme':'base',
        'themeVariables': {
            'fontSize': 8px
        }
    }
}%%
flowchart LR
  id1((outlook)) -- overcast --> id2[YES]
  id1 -- rainy --> id3[(windy)]
  id1 -- sunny --> id4[(humidity)]
  id3 -- yes --> id5[NO]
  id3 -- no --> id6[YES]
  id4 -- high --> id7[NO]
  id4 -- normal --> id8[YES]
```

:::
