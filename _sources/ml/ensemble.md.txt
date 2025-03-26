# 集成学习

&emsp;&emsp;**集成学习**其实就是学习器集成，即通过构建多个学习器完成学习任务，综合所有学习器的学习结果按特定策略生成集成学习器的学习结果。结合策略一般有：平均法、投票法以及学习法等。

```{mermaid}
---
caption: Fig 1. 集成学习概要 
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
graph LR
A[集成学习] -->B[不同类预测器同一训练集];
A[集成学习] -- 并行集成 -->C[同类预测器采样训练子集];
B-->D[硬投票法];
B-->E[软投票法];
C--子集样本放回-->F[Bagging];
C--子集样本不放回-->G[Pasting];
A-- 串行集成 -->H[训练模型集成];
H-->I[Boosting];
H-->J[Stacking];
I--调整样本权重与预测器权重-->K[Adaboost];
I--对前一预测器的结果残差训练-->L[GradientBoosting];

```

- **为什么需要集成学习**？

&emsp;&emsp;（1）**通俗一点的解释：和”三个臭皮匠抵个诸葛亮“是一个道理**。<br>&emsp;&emsp;（2）**从理论的角度：集成学习方法可以把弱学习器变成可以精确预测的强学习器**。在PAC(Probabily approximately correct)学习框架中，如果存在一个多项式的学习算法可以学习一个类（概念），并且正确率很高，那么这个类（概念）是**强可学习**的。如果存在一个多项式的学习算法可以学习一个类（概念），学习效果（正确率）仅比随机猜测略好，那么这个概念是**弱可学习**的。已有研究证明：**强可学习与弱可学习是等价的**。也就是说，在PAC学习框架下，一个概念强可学习的充要条件是这个概念是弱可学习的。那么，已经发现的“弱可学习算法”，如何提升为“强可学习算法”？答案是集成学习。

## 投票法

&emsp;&emsp;要创建一个更好的分类器，是简单的办法就是聚合每个分类器的预测，然后将得票最多的结果作为预测类别，这种大多数投票分类器被称为**硬投票分类器**。如下图所示：

```{mermaid}
---
caption: Fig 2. 多学习器集成--投票法
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
graph LR
	A((数据集)) --train--> B[逻辑回归];
	A --train--> C[SVM];
	A --train--> D[随机森林];
	A --train--> E[...];
	A --train--> F[其它分类器];
	B -.predict.-> G[+1/-1];
	C -.predict.-> H[+1/-1];
	D -.predict.-> I[+1/-1];
	E -.predict.-> J[+1/-1];
	F -.predict.-> K[+1/-1];
	G --vote--> L[+1/-1];
	H --vote--> L[+1/-1];
	I --vote--> L[+1/-1];
	J --vote--> L[+1/-1];
	K --vote--> L[+1/-1];

```

- **投票法效果**

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
#数据集
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)

#硬投票法集成
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864
    RandomForestClassifier 0.872
    SVC 0.888
    VotingClassifier 0.896

```python
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)

#软投票法
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864
    RandomForestClassifier 0.872
    SVC 0.888
    VotingClassifier 0.912


&emsp;&emsp;投票分类器的准确率通常比集成中最好的分类器还要高。事实上，即使每个分类器都是弱学习器，通过集成依然可以实现一个强学习器。只要有足够大数量且多种类的弱学习器即可。

- **为什么会有效？**

&emsp;&emsp;下面的类比可以帮助理解。假设一个略微偏倚的硬币，它有51％的机率正面向上。如果投1000次，大致会得到510次正面，所以正面是大多数。也就是说，在1000次投掷之后，大多数硬币正面向上（1000硬币参与结果投票，至少501个硬币结果为正面）的概率接近75％。即

$$
1-\textrm{pbinom}(499,1000,0.51)\approx 74.67\%
$$

10000次后，这个概率达到97%。同理，假设有1000个分类器的集成，每个分类器都只有51%的概率是正确的。如果，以大多数投票的类别作为预测结果，可以达到准确率接近75%。

## 并行集成：bagging和pasting

&emsp;&emsp;每个预测器使用相同算法，但在训练集的随机子集上进行训练。如果样本放回，这种方法称为bagging(bootstrap aggregating)；如果样本不放回，这种方法称为pasting。


```{mermaid}
---
caption: Fig 3. 并行集成示例
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
graph LR
	A[(训练集)]  --随机选择--> B[(子集1)]
	A[(训练集)]  --随机选择--> C[(子集2)]
	A[(训练集)]  --随机选择--> D[(子集3)]
	A[(训练集)]  --随机选择--> E[(...)]
	A[(训练集)]  --随机选择--> F[(子集k)]
	B --训练--> G[分类器1]
	C --训练--> H[分类器2]
	D --训练--> I[分类器3]
	E --训练--> J[...]
	 F --训练--> K[分类器k]
	 G --测试集预测--> L[+1/-1]
	 H --测试集预测-->M[+1/-1]
	 I --测试集预测--> N[+1/-1]
	J --测试集预测--> O[+1/-1]
	K --测试集预测--> P[+1/-1]
	L --集成--> Q[+1/-1]
	O--集成--> Q[+1/-1]
	M--集成--> Q[+1/-1]
	N --集成--> Q[+1/-1]
	P --集成--> Q[+1/-1]
```

- **并行集成的效果示例**

```python
#准备数据集
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Bagging集成分类器
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), #分类器种类
    n_estimators=500,   #分类器个数
    max_samples=100,    #训练子集样本数
    bootstrap=True,     #bagging=True; pasting=False
    random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

#打印结果
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#对比决策树
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
```

    0.904
    0.856

## 串行集成：提升法

&emsp;&emsp;提升法是指可以将几个弱学习器结合成一个强学习器的任意集成方法。大多数提升法的**总体思路**是循环训练预测器，每次都其前序做出一些改正。

```{mermaid}
---
caption: Fig 4. 提升法示意
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
flowchart  LR
  A2 o--o C1;
  A3 o--o C2;
  A4 o--o C3;  
  subgraph  次序训练1 
  A1[数据集0]--训练-->B1[预测器];
  B1 --调整数据权重--> C1[数据集1];
  end
  subgraph 次序训练2
  A2[数据集1]--训练-->B2[预测器];
  B2--调整数据权重--> C2[数据集2];
  end
  subgraph 次序训练...
  A3[...]-..- C3[...];
  end
  subgraph 次序训练n
  A4[数据集n-1]--训练-->B4[预测器n];
  end
```
### Adaboost

&emsp;&emsp;假设二分类情况，以下给出了Adaboost算法的框架。给定训练集$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$，其中$x_i\in\mathcal{X}\subseteq \mathbb{R}^n, y_i\in\mathcal{Y}=\{+1,-1\}$。

| 算法：Adaboost$\left(S=\{(\pmb{x}_1,y_1),(\pmb{x}_2,y_2),...,(\pmb{x}_m,y_m)\}\right)$|
| :--- |
|01&emsp;**for** $i=1$ **to** $m$<br>02&emsp;&emsp;&emsp;$w_1(i)=\frac1m$<br>03&emsp;**for** $t=1$ **to** $T$<br>04&emsp;&emsp;&emsp;$f_t\leftarrow$误差$\epsilon_t=\mathop{\mathbb{P}}\limits_{\pmb{x}_i\sim w_t}\left[f_t(\pmb{x}_i)\neq y_i\right]$较小的基分类器<br>05&emsp;&emsp;&emsp;$\alpha_t\leftarrow \frac12\log\frac{1-\epsilon_t}{\epsilon_t}$<br>06&emsp;&emsp;&emsp;$Z_t\leftarrow 2[\epsilon_t(1-\epsilon_t)]^{\frac12}$ (归一化因子)<br>07&emsp;&emsp;&emsp;**for** $i=1$ **to** $m$<br>08&emsp;&emsp;&emsp;&emsp;&emsp;$w_{t+1}(i)\leftarrow\frac{w_t(i)\exp(-\alpha_t\times y_i\times f_t(\pmb{x}_i))}{Z_t}$<br>09&emsp;$f\leftarrow \sum_{t=1}^T\alpha_t f_t$<br>10&emsp;**return** $f$ |


+ 初使化训练数据的权值分布

$$
W_1=(w^{(1)},...,w^{(N)}), w_{1i}=\frac{1}{N},i=1,2,...,N
$$

使用该权值分布对第一个预测器进行训练,

$$
f_j(x):\mathcal{X}\rightarrow \{+1,-1\}, j=1,2,...,M
$$

计算加权误差率$r_1$如下，

$$
r_j=\frac{\sum_{i=1,y_j^{(i)}\neq y^{(i)}}^N w^{(i)} }{\sum_{i=1}^N w^{(i)} }, j=1,2,...,M
$$

计算预测器权重$a_j$如下，

$$
\alpha_j=\eta\log\frac{1-r_j}{r_j}, j=1,2,...,M
$$

+ 更新权值

&emsp;&emsp;对于$i=1,2,...,N$，如果$y_j^{(i)}=y^{(i)}$，则$w^{(i)}\leftarrow w^{(i)}$；否则，

$$
w^{(i)}\leftarrow w^{(i)}\exp(\alpha_j)
$$

最后归一化，即，

$$
w^{(i)}=\frac{w^{(i)}}{\sum_{i=1}^N w^{(i)}}
$$

+ 以此规则，训练后序所有预测器。

#### AdaBoost分类器预测
&emsp;&emsp;集成分类器训练好之后，就可以按以下规则预测，

$$
\hat{y}(x)=\arg\max_{k} \sum_{j=1,\hat{y}_j(x)=k}^M \alpha_j
$$


#### AdaBoost例子
&emsp;&emsp;给定以下训练数据。假设弱分类器由$x<v$或$x>v$产生，其阈值$v$使该分类器在训练数据集上分类误差率最低。下面用AdaBoost算法学习一个强分类器。


| 序号 | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $x$    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| $y$    | 1    | 1    | 1    | -1   | -1   | -1   | 1    | 1    | 1    | -1   |

<div align="center">表1：训练数据</div>

&emsp;&emsp;第一步，先求数据集的初使权值分布，

$$
w^{(i)}=\frac{1}{10}, i=1,2,...,10
$$

在这个权值分布训练数据上，阈值取3.5时，分类误差率最低，故基本分类器为$f_1(x)=1; x<3.5$。$f_1(x)$在训练集的误差$r_1$为$0.3$。所以，分类器$f_1(x)$的权值为：$a_1=\eta\log\frac{1-r_1}{r_1}=0.4236$，这里$\eta=0.5$。

&emsp;&emsp;第二步，更新权值分布，

$$
\begin{split}
w^{(i)}&=0.1;\quad i=1,2,...,6,10.\\
w^{(i)}&=0.1\exp(a_1)=0.1527;\quad i=7,8,9.
\end{split}
$$

&emsp;&emsp;归一化后得，

| 序号 | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $w$    | 0.08633    | 0.08633    |0.08633    | 0.08633    | 0.08633    | 0.08633   | 0.13188    | 0.13188    | 0.13188    | 0.08633   |

分类器$f_1(x)$的权值$u^{(1)}$为：$0.4236$

&emsp;&emsp;第三步，以第一步和第二步的规则执行，直到所有分类器都已训练。

&emsp;&emsp;最终分类器为

$$
f(x)=\textrm{sign}(u^{(1)}f_1(x)+u^{(2)}f_2(x)+...+u^{(k)}f_k(x))
$$
