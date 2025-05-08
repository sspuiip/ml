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

### 多维缩放

&emsp;&emsp;多维缩放(multiple dimensional scaling, MDS)的**主要思想**是原始空间中样本之间的距离在低维空间得以保持。

&emsp;&emsp;假设$m$个样本在原始空间($d$维)的**距离矩阵**为$\pmb{D}\subseteq \mathbb{R}^{m\times m}$，样本集映射后在$d'$维空间的表示为$\pmb{Z}\in \mathbb{R}^{m\times d'}$。MDS的任务是获得$d'$维空间的数据矩阵$\pmb{Z}$，且任意两个样本在$d'$维空间的欧式距离$\parallel\pmb{z}_i-\pmb{z}_j\parallel^2$等于原始空间的距离$D_{ij}$，即

$$
\boxed{
\parallel \pmb{z}_i-\pmb{z}_j\parallel^2=D_{ij},\quad \forall i,j\in [1,m]. }
$$(mds-target)

- **映射前后样本距离保持一致**

&emsp;&emsp;**求解**MDS。假设映射**且中心化**后样本集$\pmb{Z}=\{\pmb{z}_i\}_{i=1}^m$的**内积矩阵**为$\pmb{B}$。根据条件，映射前后的距离要保持一致，可将等式{eq}`mds-target`左边改写为，

$$
\begin{split}
\parallel \pmb{z}_i-\pmb{z}_j\parallel^2&=\parallel\pmb{z}_i\parallel^2+\parallel\pmb{z}_j\parallel^2-2\pmb{z}_i^\top\pmb{z}_j\\
&=B_{ii}+B_{jj}-2B_{ij}\\
&=D_{ij}
\end{split}
$$(detailed-mds-target)

&emsp;&emsp;数据矩阵$\pmb{Z}$已中心化(样本=样本-样本均值)，则可得到以下结论，

$$
\begin{split}
\sum_{i}D_{ij}&=\sum_i B_{ii}+B_{jj}-2B_{ij}=\textrm{tr}(\pmb{B})+mB_{jj}\\
\sum_{j}D_{ij}&=\sum_i B_{ii}+B_{jj}-2B_{ij}=\textrm{tr}(\pmb{B})+mB_{ii}\\
\sum_{ij}D_{ij}&=\sum_{ij} B_{ii}+B_{jj}-2B_{ij}=\sum_j\textrm{tr}(\pmb{B})+mB_{jj}=2m\cdot\textrm{tr}(\pmb{B})\\
\end{split}
$$

令，

$$
\begin{split}
D_{i\cdot}&\triangleq\frac1m\sum_{j}D_{ij}\\
D_{\cdot j}&\triangleq\frac1m\sum_{i}D_{ij}\\
D_{\cdot\cdot}&\triangleq\frac{1}{m^2}\sum_i\sum_{j}D_{ij}\\
\end{split}
$$

综合上述结论，代入等式{eq}`detailed-mds-target`，可得最终结论，

$$
\boxed{
\begin{split}
B_{ij}&=-\frac12(D_{ij}-B_{ii}-B_{jj})\\
&=-\frac12\left(D_{ij}-D_{i\cdot}-D_{\cdot j}+D_{\cdot\cdot} \right)
\end{split}}
$$(mds-solution)

&emsp;&emsp;等式{eq}`mds-solution`**说明**：在保持映射前后样本距离不变的前提下，映射后的$d'$维空间样本的内积矩阵$\pmb{B}$与原$d$维空间样本的距离矩阵$\pmb{D}$的双中心化结果一致。这也是双中心化的意义，即距离信息可转化为内种形式（可表示为相似性）。

- **获得降维后的样本矩阵$\pmb{Z}$**

&emsp;&emsp;对矩阵$\pmb{B}$（实对称矩阵）做特征值分解，$\pmb{B}=\pmb{V\Lambda V}^\top=\pmb{Z}^\top\pmb{Z}$。假设有$d_*$个非零特征值构成对角矩阵$\pmb{\Lambda}_*=\textrm{diag}(\lambda_1,\lambda_2,...,\lambda_{d_*})$,以及所对应的特征向量矩阵$\pmb{V}_*$，则$\pmb{Z}$可以表示为，

$$
\pmb{Z}=\pmb{\Lambda}_*^{\frac{1}{2}}\pmb{V}_*^\top \in \mathbb{R}^{m\times d_*}
$$

&emsp;&emsp;现实应用中，可以选择$d'<d$个最大特征值构成的对角阵$\hat{\pmb{\Lambda}}$及特征向量矩阵$\hat{\pmb{V}}$，即

$$
\pmb{Z}=\hat{\pmb{\Lambda}}^{\frac{1}{2}} \hat{\pmb{V}}^\top \in \mathbb{R}^{m\times d'}
$$


#### 中心化

- **中心化**

&emsp;&emsp;所谓**中心化**是指，对所有样本减去样本均值。定义以下矩阵为中心化矩阵，即

$$
\pmb{H}\triangleq \pmb{I}_m-\frac1m\pmb{1}\pmb{1}^\top
$$(center-matrix)

&emsp;&emsp;若有矩阵$\pmb{K}\in\mathbb{R}^{m\times m}$，则中心化矩阵有3种情形：

|中心化操作|意义|
|:---:|:---:|
|$\pmb{HK}$左乘中心化矩阵|对矩阵$\pmb{K}$的每列（列向量）进行中心化。|
|$\pmb{KH}$右乘中心化矩阵|对矩阵$\pmb{K}$的每行（行向量）进行中心化。|
|$-\frac12\pmb{HKH}$双中心化|对矩阵$\pmb{K}$的每行每列进行中心化。|

- **双中心化**

&emsp;&emsp;**双中心化**：若将距离矩阵进行双中心化，则可将距离信息转化为内积形式，这与多维缩放的结果一致。假设$D_{ij}=\parallel \pmb{x}_i-\pmb{x}_j\parallel^2$，数据矩阵$\pmb{X}\in\mathbb{R}^{d\times m}$，中心化的样本$\pmb{x}_i^*=\pmb{x}_i-\bar{\pmb{x}}$，Gram矩阵$\pmb{K}=\pmb{X}^\top\pmb{X}$，则有，

$$
\begin{split}
\pmb{K}^*\triangleq{\pmb{X}^*}^\top\pmb{X}^*&=\pmb{XH}^\top\pmb{XH}\\
&=\pmb{H}\pmb{X}^\top\pmb{X}\pmb{H}\\
&=\pmb{HKH}\\
&=\left(\pmb{I}_m-\frac1m\pmb{1}\pmb{1}^\top\right)\pmb{K}\left(\pmb{I}_m-\frac1m\pmb{1}\pmb{1}^\top\right)\\
&=\pmb{K}-\frac1m\pmb{K}\pmb{1}\pmb{1}^\top-\frac1m\pmb{1}\pmb{1}^\top\pmb{K}+\frac{1}{m^2}\pmb{1}\pmb{1}^\top\pmb{K}\pmb{1}\pmb{1}^\top
\end{split}
$$

&emsp;&emsp;若对$\pmb{D}$(实对称矩阵)进行双中心化(先中心化再乘上因子$-\frac12$)，则正好有以下结论，

$$
\begin{split}
\pmb{D}^*&=\pmb{HDH}\\
&=\pmb{D}-\frac1m\pmb{D}\pmb{1}\pmb{1}^\top-\frac1m\pmb{1}\pmb{1}^\top\pmb{D}+\frac{1}{m^2}\pmb{1}\pmb{1}^\top\pmb{D}\pmb{1}\pmb{1}^\top
\end{split}
$$

即

$$
\boxed{
-\frac12\pmb{HDH}=\pmb{B}}
$$

&emsp;&emsp;这说明**双中心化与多维缩放的结论是一致的**。

#### 算法

|算法：多维缩放|
|:---|
|&emsp;&emsp;**输入**：距离矩阵$\pmb{D}$，低维空间维数$d'$.<br/>&emsp;&emsp;**过程**：<br/>&emsp;&emsp;&emsp;&emsp;1. 计算$\pmb{D}$;<br/>&emsp;&emsp;&emsp;&emsp;2. 计算矩阵$\pmb{B}$;<br/>&emsp;&emsp;&emsp;&emsp;3. 矩阵$\pmb{B}$做特征值分解；<br/>&emsp;&emsp;&emsp;&emsp;4. 选取$\hat{\pmb{V}},\hat{\pmb{\Lambda}}$；<br/>&emsp;&emsp;**输出**： 矩阵$\hat{\pmb{V}}\hat{\pmb{\Lambda}}^{1/2}$每一行即为一个样本的低维坐标。|


### 等度量映射

&emsp;&emsp;等度量映射(Isometric Mapping, Isomap)的**基本出发点**在于，Isomap认为低维流行嵌入到高维空间之后，直接在高维空间计算直线距离具有误导性，因为高维空间的直线距离在低维流行是不可达的（如：瑞士卷上两个点（位于同一$x,y$坐标，$z$不同坐标）是不能用直线距离来计算的，因为该流行是扭曲过的）。**流形学习**认为$d$维流形是$n$维空间$(d<n)$的一部分，局部类似于$d$维超平面。例如：2D流形是一个2D形状，该形状可以在更高维的空间中弯曲和扭曲。因此，低维嵌入流形上的**本真距离**（**测地线距离**）不能用高维空间的直线距离来计算，但能用近邻距离来近似。

&emsp;&emsp;**如何计算测地线距离呢**？利用流形在局部与欧氏空间同胚这个性质，对每个样本点基于欧氏距离找出其近邻点，建立一个近邻连接图。于是，计算两点之间的测地线距离的问题就转变为计算近邻连接图上两点之间最短路径的问题。近邻图计算两点之间的最短路径，可以采用Dijkstra算法或Floyd算法，在得到任意两点的距离之后，就可以用多维缩放(MDS)方法来获得样本点在低维空间的坐标。

#### Isomap算法

|算法：Isometric Mapping, Isomap|
|:---|
|**输入**：样本集$\mathcal{D}=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$，低维空间维数$d'$.<br/>**过程**：<br/>&emsp;&emsp;1. 确定每个样本$\pmb{x}_i$的$k$近邻;<br/>&emsp;&emsp;2. 使用最短路径算法(例如：Dijkstra)计算$k$近邻图的任意样本间距离$dist(\pmb{x}_i,\pmb{x}_j)$;<br/>&emsp;&emsp;3. 以$dist(\pmb{x}_i,\pmb{x}_j)$为输入，使用MDS计算低维坐标；<br/>**输出**： MDS计算的低维坐标。|


#### 流形

&emsp;&emsp;在介绍流形前，需要一些有关的背景知识。

- **拓扑空间**

&emsp;&emsp;给定集合$\mathcal{X}$，以及$\mathcal{X}$的一些子集构成的族$\mathcal{O}$，如果以下性质成立，则$(\mathcal{X},\mathcal{O})$称为一个拓扑空间：
1. $\emptyset$和$\mathcal{X}$都属于$\mathcal{O}$；
2. $\mathcal{O}$中的任意多个元素的并仍属于$\mathcal{O}$；
3. $\mathcal{O}$中的任意多个元素的交仍属于$\mathcal{O}$；

此时，$\mathcal{X}$中的元素称为点，$\mathcal{O}$中的元素称为开集（可以理解为开区间）。


- **度量空间**

&emsp;&emsp;度量空间是一个二元对$(\mathcal{M},d)$，其中$\mathcal{M}$是一个集合，$d$是定义在$\mathcal{M}$上的一个度量，即映射$d: \mathcal{M}\times \mathcal{M}\rightarrow \mathbb{R}$，对于任意$\pmb{x,y,z}\in M$满足以下条件：
1. $d(\pmb{x,y})=0 \Leftrightarrow \pmb{x}=\pmb{y}$;
2. $d(\pmb{x,y})=d(\pmb{y,x})$;
3. $d(\pmb{x,z})\le d(\pmb{x,y})+d(\pmb{y,z})$;

- **流形**

&emsp;&emsp;流形是一个拓扑空间，对于每个点，其周围的邻域局部类似于欧几里得空间。更确切地说，$n$维流形的每个点都有一个邻域开集，该邻域与$n$维欧几里德空间的邻域开集同胚。人们经常可以想象拉伸或平坦流形的局部邻域以得到一个平坦的欧几里得平面。大致地说，拓扑空间是一个几何物体，同胚就是把物体连续延展和弯曲，使其成为一个新的物体。因此，正方形和圆是同胚的，但球面和环面就不是。

&emsp;&emsp;在拓扑学中，同胚(homeomorphism、topological isomorphism、bi continuous function)是两个拓扑空间之间的**双连续函数**。同胚是拓扑空间范畴中的同构；也就是说，它们是保持给定空间的所有拓扑性质的映射。如果两个空间之间存在同胚，那么这两个空间就称为同胚的，从拓扑学的观点来看，两个空间是相同的。

&emsp;&emsp;一个较大的$m$维空间($n<m$)中的$n$维流形
)局部类似于$n$维欧几里得超平面。例如

1. 1维流形：圆，正方形，曲线等。但8字形不是1维流形，因为8字的中心点局部是2维的欧氏空间同胚。
2. 2维流形：球面，环面等。

&emsp;&emsp;由于流形结构是由“局部”类似于欧几里得空间的性质定义的，我们不必考虑任何全局的、外部定义的坐标系的几何关系，相反，我们可以只考虑流形的内在几何和拓扑性质。

### 拉普拉斯特征映射

&emsp;&emsp;拉普拉斯特征映射(Laplacian eigenmaps)的基本思想是：保留高维数据的局部结构。即高维空间的两个样本是近邻，则在低维中也应是近邻。因此，Laplacian eigenmaps的目标是通过高维空间权重矩阵$\pmb{W}$衡量的近邻关系最大化的保留到低维空间。