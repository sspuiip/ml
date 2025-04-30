# 数据降维

&emsp;&emsp;现实应用中，经常会遇到大量的高维数据。一方面来说，随着维度的增加，高维样本可以使得样本之间的区别更加显著。例如：只提供身高信息判断该样本是否为男性的判别问题中，如果仅依赖身高这个维度，则会有较大概率判断失误。在身高的基础之上，如果再添加头发长度、喉结、胸围等特征，则有较大概率判断正确。然而，维度的增加也不总是合理的。从另外一方面来说，若增加的维度超出了临界值高维空间也会还来一些不良的后果。例如：样本对之间的距离会迅速增大（稀疏）、距离不再具有区分性（所有点对之间都差不多远）、区别样本所需的数量呈指数级增长才能覆盖空间等。这些特性也称之为**维度灾难**。

&emsp;&emsp;为了避免维度灾难，以及找到问题求解最合适的数据表示形式，需要研究原有数据的表示问题（合适的维度），这一过程也称之为**数据降维**。

## 主成分分析

&emsp;&emsp;主成分分析(Principal Component Analysis, PCA)是一种通过某种正交变换将一组可能存在相关关系的变量（$n$个维度为$n$变量）转换为一组线性不相关的变量（降维后的$d$个维度）。若有训练数据（$n$：维度，$m$：样本数），

$$
\pmb{X}=\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_m\\ |&|&\dots&| \end{pmatrix}_{n\times m}
$$

其中，$\pmb{x}_i=(x_{i1},...,x_{in})^\top$。PCA的**目标**是找到一个基$(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_d)=\pmb{W}_{n\times d}$，使得变换后样本集$\pmb{Z}=\pmb{W}^\top\pmb{X}$的重构矩阵$\hat{\pmb{X}}=\pmb{WZ}$与原数据矩阵$\pmb{X}$的误差尽可能的小，即$\pmb{X}$与$\hat{\pmb{X}}$的误差最少。也就是说，投影的超平面$\pmb{W}$使得投影后的数据矩阵$\pmb{Z}$丢失的信息最少。

&emsp;&emsp;如何找到这个投影超平面$\pmb{W}$呢？一个可行的办法是比较$\pmb{X}$与$\hat{\pmb{X}}$之间的平均距离（$\frac1m\sum_i^m\parallel \pmb{x}_i-\hat{\pmb{x}}_i\parallel^2$），使得这个距离最小的超平面就是最优投影超平面。这是**PCA的主要思想**。

&emsp;&emsp;假设数据样本已<font color="red">中心化</font>(所有样本减去均值即为中心化；中心化之后再除以样本标准差即为标准化)，变换后的新坐标系为$(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_d)$，若丢弃部分坐标，将维度降至$d'<d$，则样本在低维坐标系中的投影为$\pmb{z}_i=(z_{i1},z_{i2},...,z_{id'})$ ，其中$z_{ij}=\langle \pmb{x}_i,\pmb{w}_j\rangle$是$\pmb{x}_i$在低维坐标系的第$j$维的坐标。若用$\pmb{z}_i$来重构$\pmb{x}_i$，则会有$\hat{\pmb{x}}_i=\sum_{j=1}^{d'}z_{ij}\pmb{w}_j=\pmb{W}\pmb{z}_i$。

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
$$(pca-target)

其中，

$$
\pmb{X}=\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_m\\ |&|&\dots&| \end{pmatrix}_{d\times m}, \qquad
\pmb{W}=\begin{pmatrix}|&|&\dots&|\\\pmb{w}_1&\pmb{w}_2&\dots&x_{d'}\\ |&|&\dots&| \end{pmatrix}_{d\times d'}
$$

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
$$(pca-solution)

于是，只要对样本协方差矩阵进行特征值分解，将求得的特征值排序后，取前$d'$个特征值对应的特征向量构成<font color="red">投影矩阵$W^*=(w_1,w_2,...,w_{d'})$</font>。该矩阵即为主成分分析的**解**。


### PCA算法1--特征值分解


&emsp;&emsp;通过样本协方差矩阵计算PCA。

|算法：特征值分解p实现PCA|
|:--|
|**输入**：样本集$\mathcal{D}=\{x_1,x_2,...,x_m\}$，低维空间维数$d'$.<br/>**过程**：<br/>&emsp;&emsp;1：样本中心化： $\pmb{x}_i=\pmb{x}_i-\frac{1}{m}\sum_{i=1}^m\pmb{x}_i$；<br/>&emsp;&emsp;2：计算样本的协方差矩阵$\pmb{XX}^T$;<br/>&emsp;&emsp;3：对协方差矩阵做特征值分解；<br/>&emsp;&emsp;4：取出最大的$d'$个特征值对应的特征向量$\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_{d'}$；<br/>**输出**： 投影矩阵$\pmb{W}^*=(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_{d'})$。|


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
    
    for xy in zip(X[:,0],X[:,1]):
            plt.annotate("(%.0f,%.0f)"%(xy[0],xy[1]), xy, xytext=(-10,10), textcoords='offset points') #标注数据样本
    plt.axis('equal')
```

### PCA算法2--SVD分解

&emsp;&emsp;PCA除了对于协方差矩阵$\pmb{XX}^\top$进行特征值分解计算得到投影特征向量之外，还可以通过SVD矩阵分解技术得到投影向量。SVD矩阵分解如下式所示，

$$
\hat{\pmb{X}}=\pmb{U\Sigma V}^\top, \quad \hat{\pmb{X}}^\top=\pmb{V\Sigma U}^\top
$$

这里的$\hat{\pmb{X}}$是正常的数据集矩阵。即

$$
\hat{\pmb{X}}=\begin{pmatrix}-&\pmb{x}_1 &-\\ -&\pmb{x}_1 &-\\ 
\vdots&\vdots &\vdots\\ -&\pmb{x}_m &-\\ \end{pmatrix}
$$


&emsp;&emsp;PCA中的$\pmb{X}=\hat{\pmb{X}}^\top$，因此有，

$$
\pmb{XX}^\top=\hat{\pmb{X}}^\top\hat{\pmb{X}}=\pmb{V\Sigma U}^\top \pmb{U\Sigma V}^\top=\pmb{V\Sigma}^2\pmb{V}^\top
$$

&emsp;&emsp;最终，$\pmb{V}$的最大前$d'$个特征值对应的特征向量所组成的矩阵即为变换矩阵$\pmb{W}^*=(\pmb{w}_1,\pmb{w}_2,...,\pmb{w}_{d'})$。而$\pmb{V}$可以通过SVD分解获得。

&emsp;&emsp;投影后的新数据($d'<n$)，

$$
\pmb{Z}=\pmb{X}^\top\pmb{W}^*=\pmb{U\Sigma V}^\top\pmb{W}^*\approx\pmb{U\Sigma}
$$(pca-svd-proj)

+ 示例

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
    features=np.transpose(features)
    data=np.dot(X,features)
    return data,features
def pca_svd(X,k):
    X=X-X.mean(axis=0)
    U,S,Vt=np.linalg.svd(X)
    W=Vt.T[:,:k]
    data = np.dot(X,W)
    return data,W
if __name__=="__main__":
    
    X = np.array([[-1, -1.5], [-2, -1], [-3, -2], [1, 2], [2, 1], [3, 2],[1,3],[-1.5,1]]) 
    X_new,features=pca(X,2)
    X_new_svd,f_svd=pca_svd(X, 1)
    print('X_eig_decom',X_new)
    print('X_svd',X_new_svd)
    
    
    for i in range(2):    
        a,b=features[:,i]    
        plt.plot(X[:,0],X[:,1],'ro')#,c = 'r',marker = 'o')
        y=np.linspace(-6,6,10)
        x1=y*np.cos(np.arctan(b/a))#features[1,0]/features[0,0]))
        y1=y*np.sin(np.arctan(b/a))#features[1,0]/features[0,0]))
        plt.plot(x1,y1,'b--')    
    
        proj_dir=np.array([a,b])#features[0,0],features[1,0]])
        proj_dir=proj_dir/np.linalg.norm(proj_dir)
        #计算投影
        PX=[]
        for x in X:
            p=np.dot(x,proj_dir)/np.linalg.norm(proj_dir)
            px=p*proj_dir
            PX.append(px)
            #px=p*np.cos(np.arctan(features[1,0]/features[0,0]))
            #py=p*np.sin(np.arctan(features[1,0]/features[0,0]))
            #PX.append([px,py])
        PX=np.array(PX)  
        for ix in range(X.shape[0]):
            plt.scatter(PX[ix,0],PX[ix,1],c='b',marker="s")
            plt.plot([X[ix,0],PX[ix,0]],[X[ix,1],PX[ix,1]],'y:')       
        
   
    plt.axis('equal')

```

### 核主成分分析

&emsp;&emsp;前面我们通过计算样本协方差矩阵$\pmb{XX}^\top$的特征向量组成投影矩阵来实现PCA。对于核函数的隐式映射$\phi :\pmb{x}\rightarrow \phi(\pmb{x})$形成的映射数据矩阵$\pmb{\Phi}^\top$，如何计算PCA。也就是映射后的协方差矩阵$\pmb{\Phi\Phi}^\top$如何分解出特征向量组成投影矩阵？针对这一问题，研究人员提出了核主成分分析(kernel PCA)。

&emsp;&emsp;**首先考查核矩阵$\pmb{K}\triangleq\pmb{X}^\top\pmb{X}$与协方差矩阵$\pmb{C}\triangleq\frac1m\pmb{XX}^\top$特征向量之间的关系**。
对实对称矩阵$\pmb{X}^\top\pmb{X}$进行特征值分解$\pmb{X}^\top\pmb{X}\pmb{U}=\pmb{U\Lambda}$，等式两边同时乘上$\pmb{X}$，则可以得到，

$$
(\pmb{XX}^\top)(\pmb{XU})=(\pmb{XU})\pmb{\Lambda}
$$

从上式可以得到$\pmb{XX}^\top$的特征向量为$\pmb{V}\triangleq\pmb{XU}$，特征值对角矩阵为$\pmb{\Lambda}$。注意到特征向量的模长，

$$
\Vert \pmb{v}_j\Vert^2=\pmb{u}_j^\top\pmb{X}^\top\pmb{X}\pmb{u}_j=\pmb{u}_j^\top\pmb{u}_j\lambda_j\pmb{u}_j^\top\pmb{u}_j=\lambda_j
$$

可以得到单位化的特征向量矩阵$\pmb{V}_{\textrm{pca}}=(\pmb{XU})\pmb{\Lambda}^{-1/2}$。

&emsp;&emsp;**现在考虑Gram矩阵$\pmb{K}\triangleq\pmb{X}^\top\pmb{X}$**。根据Mercer定理，当使用一个核函数时，隐含了一个潜在的特征空间，因此，可以将$\pmb{x}_i$表示为$\pmb{\phi}_i\triangleq\phi(\pmb{x}_i)$。相应地，数据矩阵$\pmb{X}^\top$映射为$\pmb{\Phi}^\top$，协方差矩阵$\pmb{X}\pmb{X}^\top$映射为$\pmb{\Phi}\pmb{\Phi}^\top$。由$\pmb{X}^\top\pmb{X}$与$\pmb{XX}^\top$的关系可知，$\pmb{\Phi}\pmb{\Phi}^\top$的特征向量矩阵为

$$\pmb{V}_{\textrm{kpca}}=\pmb{\Phi U\Lambda}^{-1/2}$$

其中$\pmb{U\Lambda}$分别为$\pmb{K}=\pmb{\Phi}^\top\pmb{\Phi}$的特征向量矩阵以及对应的特征值。

&emsp;&emsp;根据上面计算的结果，从特征向量矩阵中取$k$个特征向量即可组成投影矩阵，经过数据投影即可得到样本的$k$维压缩表示。**但是**，映射$\phi()$可能没有显示表示，或难以直接计算。**解决办法是使用核函数间接计算$\phi()$**。任意给定样本$\pmb{x}_*$，则其在特征空间的投影$\hat{\pmb{x}}_i$可通过以下方式计算。

$$
\hat{\pmb{x}}_i=\phi(\pmb{x}_*)^\top\pmb{V}_{\textrm{kpca}}=\phi(\pmb{x}_*)^\top\pmb{\Phi U\Lambda}^{-1/2}=\pmb{k}_*^{\top}\pmb{U\Lambda}^{-1/2}
$$

&emsp;&emsp;最后要注意的是$\pmb{K}$在特征值分解之前，需要中心化。中心化可通过以下步骤计算得到。

$$
\begin{split}
\tilde{\pmb{K}}&=\pmb{K}-\frac1N\pmb{K11}^\top-\frac1N\pmb{11}^\top\pmb{K}+\frac{1}{N^2}(\pmb{1}^\top\pmb{K}\pmb{1})\pmb{11}^\top\\
&=\pmb{K}-\pmb{KO}-\pmb{{OK}}+\pmb{OKO}
\end{split}
$$

其中，$\pmb{O}=\frac1N \pmb{1}\pmb{1}^\top, \pmb{1}=[1,1,...,1]_{1\times N}^\top$。

- 示例

```python
def kpca(X,k):
    """ 

    Parameters
    ----------
    X : np.array with n x d
    k : int rank of the low-dimension

    Returns
    -------
    data : projection data

    """
    n,d = X.shape
    if d < k:
        print("\nDimensions of output data has to be lesser than the dimensions of input data\n")
        return
    
    # construct K
    K = np.zeros((n,n))
    for row in range(n):
        for col in range(row+1):
            k_ij = np.sum((X[row,:]-X[col,:])**2)
            K[row,col]=np.exp(-k_ij)
    K = K+K.T
    for row in range(n):
        K[row,row]=K[row,row]/2
        
    # normalize K
    all1 = np.ones((n,n))/n
    K_center = K - np.dot(all1,K)-np.dot(K,all1)+np.dot(np.dot(all1,K),all1)
    
    # eigvector
    S,U= np.linalg.eig(K_center)      
    V=np.dot(U,np.diag(1/np.sqrt(np.abs(S))))
    
    eig_pairs=[(S[i],V[:,i]) for i in range(len(S))]
    eig_pairs.sort(reverse=True)
    
    V = np.array(([ele[1] for ele in eig_pairs[:k]]))
    V = V.T
    data=np.dot(K_center,V)
    return data
```
