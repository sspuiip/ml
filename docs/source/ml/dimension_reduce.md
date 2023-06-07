# 表示学习

&emsp;&emsp;一般来说，机器学习中的原始数据是高维数据。高维数据往往具有复杂性、冗余性等特点。高维空间会有很多不一样的特征，也称之为维度灾难。

- 大多数数据对象之间相距都很远(高维数据有很大可能非常稀疏)

- 理论上，通过增大训练集，使训练集达到足够密度，是可以避免维度灾难的。但实践中，要达到给定密度，所需的训练数据随着维度的增加呈指数式上升

&emsp;&emsp;为了避免维度灾难，以及找到问题求解最合适的数据表示形式，需要研究原有数据的表示问题，这一过程也称之为**表示学习**。


## 主成分分析

&emsp;&emsp;主成分分析(Principal Component Analysis, PCA)是一种通过某种正交变换将一组可能存在相关关系的变量转换为一组线性不相关的变量。对于训练数据，

$$
\pmb{X}=\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_m\\ |&|&\dots&| \end{pmatrix}_{n\times m}
$$

其中，$\pmb{x}_i=(x_{i1},...,x_{in})^\top$。PCA的**目标**是找到一个基$(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_d)=\pmb{W}_{n\times d}$，使得$\pmb{Z}=\pmb{W}^\top\pmb{X}$的重构矩阵$\hat{\pmb{X}}=\pmb{WZ}$与$\pmb{X}$的误差尽可能的小，即，投影的超平面$\pmb{W}$使得投影后的数据矩阵$\pmb{Z}$丢失的信息最少。

&emsp;&emsp;如何找到这个超平面呢？可行的一个办法是比较$\pmb{X}$与$\hat{\pmb{X}}$之间的平均距离（$\parallel \pmb{x}_i-\hat{\pmb{x}}_i\parallel^2$），使得这个距离最小的超平面就是最优投影超平面。这是PCA这主要思想。

&emsp;&emsp;对于空间的所有样本点，如何用一个超平面来恰当的表示？有两种办法，即

+ 样本点到这个超平面的距离都足够近(投影距离最小)

+ 样本点在这个超平面的投影尽可能分开

### 最近重构性

&emsp;&emsp;假设数据样本已<font color="red">中心化</font>所有样本减去均值即为中心化；(中心化之后再除以样本标准差即为标准化)，变换后的新坐标系为$(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_d)$，若丢弃部分坐标，将维度降至$d'<d$，则样本在低维坐标系中的投影为$\pmb{z}_i=(z_{i1},z_{i2},...,z_{id'})$ ，其中$z_{ij}=\langle \pmb{x}_i,\pmb{w}_j\rangle$是$\pmb{x}_i$在低维坐标系的第$j$维的坐标。若用$\pmb{z}_i$来重构$\pmb{x}_i$，则会有$\hat{\pmb{x}}_i=\sum_{j=1}^{d'}z_{ij}\pmb{w}_j=\pmb{W}\pmb{z}_i$。

&emsp;&emsp;对于整个数据集，原样本点$\pmb{x}_i$与投影重构$\hat{\pmb{x}}_i$之间距离为，

$$
\begin{split}
\sum_{i=1}^m \left\Vert \sum_{j=1}^{d'}z_{ij}\pmb{w}_j-\pmb{x}_i\right\Vert^2 &=\sum_{i=1}^m \pmb{z}_i^\top\pmb{z}_i-2\sum_{i=1}^m\pmb{z}_i^\top\pmb{W}^\top\pmb{x}_i + \textrm{const}\\
&=\sum_{i=1}\pmb{x}_i^\top\pmb{W}\pmb{W}^\top\pmb{x}_i-2\sum_{i=1}\pmb{x}_i^\top\pmb{W}\pmb{W}^\top\pmb{x}_i +\textrm{const}\\
&=-\text{tr}\left( \sum_{i=1}\pmb{x}_i^\top\pmb{W}\pmb{W}^\top\pmb{x}_i  \right)+\textrm{const}\\
&\propto-\text{tr}\left(\pmb{W}^\top\left(\sum_{i=1}^m \pmb{x}_i\pmb{x}_i^\top  \right)\pmb{W} \right)
\end{split}
$$

PCA的优化目标则变成，

$$
\begin{split}
\min\limits_{\pmb{W}}\quad &-\text{tr}\left(\pmb{W}^\top\pmb{XX}^\top\pmb{W} \right)\\
\textrm{s.t.}\quad &\pmb{W}^\top\pmb{W}=\pmb{I}
\end{split}
$$

其中，

$$
\pmb{X}=\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_m\\ |&|&\dots&| \end{pmatrix}_{d\times m}, \qquad
\pmb{W}=\begin{pmatrix}|&|&\dots&|\\\pmb{w}_1&\pmb{w}_2&\dots&x_{d'}\\ |&|&\dots&| \end{pmatrix}_{d\times d'}
$$

### 最大可分性

&emsp;&emsp;样本点$\pmb{x}_i$在新空间的超平面投影为$\pmb{W}^\top\pmb{x}_i$，若要投影尽可能分开，则应使投影后的样本方差最大化，即，

$$
\begin{split}
\max_{\pmb{W}} \quad &\text{tr}(\pmb{W}^\top\pmb{XX}^\top\pmb{W})\\
\text{s.t.}\quad &\pmb{W}^\top\pmb{W}=\pmb{I}
\end{split}
$$

可以看出，该问题与第一种情况是等价的。

### 优化问题的求解

&emsp;&emsp;使用拉格朗日乘子法，可得，

$$
\begin{split}
\mathcal{L}(\pmb{W},\pmb{\lambda})&=\textrm{tr}(\pmb{W}^\top\pmb{XX}^\top\pmb{W})+\lambda(\pmb{W}^\top\pmb{W}-\pmb{I})\\
&=\sum_{i=1}\pmb{w}_i^\top\pmb{XX}^\top\pmb{w}_i + \sum_{i=1}\lambda_i(\pmb{w}_i^\top\pmb{w}_i-1)
\end{split}
$$

则有，$\frac{\partial \mathcal{L}}{\partial \pmb{w}_i}$,

$$
\frac{\partial \mathcal{L}}{\partial \pmb{w}_i}=\pmb{XX}^\top\pmb{w}_i-\lambda_i\pmb{w}_i
$$

令$\frac{\partial \mathcal{L}}{\partial \pmb{w}_i}=0$，则有

$$
\pmb{X}\pmb{X}^\top\pmb{w}_i=\lambda_i\pmb{w}_i
$$

于是，只要对样本协方差矩阵进行特征值分解，将求得的特征值排序后，取前$d'$个特征值对应的特征向量构成<font color="red">投影矩阵$W^*=(w_1,w_2,...,w_{d'})$</font>。该矩阵即为主成分分析的解。


### 算法


&emsp;&emsp;通过样本协方差矩阵计算PCA。

&emsp;&emsp;**输入**：样本集$\mathcal{D}=\{x_1,x_2,...,x_m\}$，低维空间维数$d'$.

&emsp;&emsp;**过程**：

&emsp;&emsp;1：样本中心化： $\pmb{x}_i=\pmb{x}_i-\frac{1}{m}\sum_{i=1}^m\pmb{x}_i$；

&emsp;&emsp;2：计算样本的协方差矩阵$\pmb{XX}^T$;

&emsp;&emsp;3：对协方差矩阵做特征值分解；

&emsp;&emsp;4：取出最大的$d'$个特征值对应的特征向量$\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_{d'}$；

&emsp;&emsp;**输出**： 投影矩阵$\pmb{W}^*=(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_{d'})$。


- **示例代码**

```python
import numpy as np
import matplotlib.pyplot as plt


def pca(X,k):
    n_samples,n_features=X.shape
    X=X-X.mean(axis=0)
    scatter_matrix=np.dot(np.transpose(X),X)
    eig_val,eig_vec=np.linalg.eig(scatter_matrix)
    eig_pairs=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    features=np.array([ele[1] for ele in eig_pairs[:k]])
    data=np.dot(X,np.transpose(features))
    return data,features

if __name__=="__main__":
    
    X = np.array([[-1, -1.5], [-2, -1], [-3, -2], [1, 2], [2, 1], [3, 2],[1,3],[-1.5,1]]) 
    X_new,features=pca(X,1)
    print(X_new)
    print(features)
    
    
    
    plt.plot(X[:,0],X[:,1],'ro')#,c = 'r',marker = 'o')
    y=np.linspace(-6,6,10)
    x1=y*np.cos(np.arctan(features[0,1]/features[0,0]))
    y1=y*np.sin(np.arctan(features[0,1]/features[0,0]))
    plt.plot(x1,y1,'b-')

    proj_dir=np.array([features[0,0],features[0,1]])
    proj_dir=proj_dir/np.linalg.norm(proj_dir)
    #计算投影
    PX=[]
    for x in X:
        p=np.dot(x,proj_dir)/np.linalg.norm(proj_dir)
        px=p*np.cos(np.arctan(features[0,1]/features[0,0]))
        py=p*np.sin(np.arctan(features[0,1]/features[0,0]))
        PX.append([px,py])
    PX=np.array(PX)  
    for ix in range(X.shape[0]):
        plt.scatter(PX[ix,0],PX[ix,1],c='b',marker="s")
        plt.plot([X[ix,0],PX[ix,0]],[X[ix,1],PX[ix,1]],'y:')
    plt.axis('equal')
```
