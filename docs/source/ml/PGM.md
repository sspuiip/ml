# 概率图模型

&emsp;&emsp;根据已观察到的数据（样本集）对未知变量（样本所属类别）进行估计，这是从数据学习知识的基本途径，也是机器学习的主要任务。概率模型将这一任务转换为计算变量的概率分布，即利用已知变量推测未知变量，也称之为**推断**。具体来说，假设所关心的变量集为$Y$，可观测变量集为$O$，其它变量为$R$，推断就是通过联合分布$P(Y,R,O)$或条件分布$P(Y,R|O)$计算得到条件分布$P(Y|O)$。其中，联合分布$P(Y,R,O)$称为**生成式模型**，条件分布$P(Y,R|O)$称为**判别式模型**。但是，直接使用概率求和规则消去变量$R$是不可行的，因为即使所有变量只有2种取值的特殊情况，计算复杂度也高达$O(2^{|Y|+|R|})$。

&emsp;&emsp;概率图模型是一种用图来表示变量间关系的概率模型。该图的结点表示一个（组）随机变量，结点之间的边表示变量间的相关关系。

```{mermaid}
mindmap
    id1["`**Root** with
a second line
Unicode works too: 🤓`"]
      id2["`The dog in **the** hog... a *very long text* that wraps to a new line`"]
      id3[Regular labels still works]
```

```{mermaid}
flowchart TB
    c1-->a2
    subgraph one
    a1-->a2
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
```

```{mermaid}
timeline
    title History of Social Media Platform
    2002 : LinkedIn
    2004 : Facebook
         : Google
    2005 : Youtube
    2006 : Twitter
```
