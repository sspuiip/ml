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

- **为什么会有效**

&emsp;&emsp;下面的类比可以帮助理解。假设一个略微偏倚的硬币，它有51％的机率正面向上。如果投1000次，大致会得到510次正面，所以正面是大多数。也就是说，在1000次投掷之后，大多数硬币正面向上（1000硬币参与结果投票，至少501个硬币结果为正面）的概率接近75％。即

$$
1-\textrm{pbinom}(499,1000,0.51)\approx 74.67\%
$$

10000次后，这个概率达到97%。
