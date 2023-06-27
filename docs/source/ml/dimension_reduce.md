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


### 算法1--特征值分解


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
    
    for xy in zip(X[:,0],X[:,1]):
            plt.annotate("(%.0f,%.0f)"%(xy[0],xy[1]), xy, xytext=(-10,10), textcoords='offset points') #标注数据样本
    plt.axis('equal')
```

### 算法2--SVD分解

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
$$

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

&emsp;&emsp;首先考查$\pmb{X}^\top\pmb{X}$与$\pmb{XX}^\top$特征向量之间的关系。
对实对称矩阵$\pmb{X}^\top\pmb{X}$进行特征值分解$\pmb{X}^\top\pmb{X}\pmb{U}=\pmb{U\Lambda}$，等式两边同时乘上$\pmb{X}$，则可以得到，

$$
(\pmb{XX}^\top)(\pmb{XU})=(\pmb{XU})\pmb{\Lambda}
$$

从上式可以得到$\pmb{XX}^\top$的特征向量为$\pmb{V}\triangleq\pmb{XU}$，特征值对角矩阵为$\pmb{\Lambda}$。注意到特征向量的模长，

$$
\Vert \pmb{v}_j\Vert^2=\pmb{u}_j^\top\pmb{X}^\top\pmb{X}\pmb{u}_j=\lambda_j\pmb{u}_j^\top\pmb{u}_j=\lambda_j
$$

可以得到单位化的特征向量矩阵$\pmb{V}_{\textrm{pca}}=(\pmb{XU})\pmb{\Lambda}^{-1/2}$。

&emsp;&emsp;现在考虑Gram矩阵$\pmb{K}\triangleq\pmb{X}^\top\pmb{X}$。根据Mercer定理，当使用一个核函数时，隐含了一个潜在的特征空间，因此，可以将$\pmb{x}_i$表示为$\pmb{\phi}_i\triangleq\phi(\pmb{x}_i)$。相应地，数据矩阵$\pmb{X}^\top$映射为$\pmb{\Phi}^\top$，协方差矩阵$\pmb{X}\pmb{X}^\top$映射为$\pmb{\Phi}\pmb{\Phi}^\top$。由$\pmb{X}^\top\pmb{X}$与$\pmb{XX}^\top$的关系可知，$\pmb{\Phi}\pmb{\Phi}^\top$的特征向量矩阵为

$$\pmb{V}_{\textrm{kpca}}=\pmb{\Phi U\Lambda}^{-1/2}$$

其中$\pmb{U\Lambda}$分别为$\pmb{K}=\pmb{\Phi}^\top\pmb{\Phi}$的特征向量矩阵以及对应的特征值。

&emsp;&emsp;根据上面计算的结果，从特征向量矩阵中取$k$个特征向量即可组成投影矩阵，经过数据投影即可得到样本的$k$维压缩表示。**但是**，映射$\phi()$可能没有显示表示，或难以直接计算。解决办法是使用核函数间接计算$\phi()$。任意给定样本$\pmb{x}_*$，则其在特征空间的投影$\hat{\pmb{x}}_i$可通过以下方式计算。

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



## 多维缩放

&emsp;&emsp;多维缩放(multiple dimensional scaling, MDS)的主要思想是原始空间中样本之间的距离在低维空间得以保持。假设$m$个样本在原始空间的距离矩阵为$\pmb{D}\subseteq \mathbb{R}^{m\times m}$。MDS的任务是获得样本集在$d'$维空间的表示$\pmb{Z}\in \mathbb{R}^{m\times d'}$，且任意两个样本在$d'$维空间的欧式距离等于原始空间的距离，即$\parallel \pmb{z}_i-\pmb{z}_j\parallel^2=$$D_{ij}, \forall 0<i,j\leq m$。

- MDS的求解

&emsp;&emsp;令$\pmb{B}=\pmb{Z}^\top \pmb{Z}\in\mathbb{R}^{m\times m}$为降维后的样本内积矩阵，$b_{ij}=\pmb{z}_i^\top\pmb{z}_j$,则有，

$$
\begin{split}
dist_{ij}^2&=\parallel z_i \parallel^2+\parallel z_j\parallel^2-2z_i^Tz_j\\
&=b_{ii}+b_{jj}-2b_{ij}\\
\end{split}
$$

假设$\pmb{Z}$已中心化，即$\sum_{i=1}^m\pmb{z}_i=0$，显然$\sum_{i=1}^mb_{ij}=\sum_{j=1}^mb_{ij}=0$，由此可知，

$$
\begin{split}
\sum_{i=1}^mdist_{ij}^2&=\text{tr}(B)+mb_{jj}\\
\sum_{j=1}^mdist_{ij}^2&=\text{tr}(B)+mb_{ii}\\
\sum_{i=1}^m\sum_{j=1}^mdist_{ij}^2&=2m\cdot\text{tr}(B)\\
\end{split}
$$

令,

$$
\begin{split}
dist_{i\cdot}^2&=\frac{1}{m}\sum_{j=1}^mdist_{ij}^2\\
dist_{\cdot j}^2&=\frac{1}{m}\sum_{i=1}^mdist_{ij}^2\\
dist_{\cdot\cdot}^2&=\frac{1}{m^2}\sum_{i=1}\sum_{j=1}^mdist_{ij}^2\\
\end{split}
$$

最终可得，

$$
b_{ij}=-\frac{1}{2}(dist_{ij}^2-dist_{i\cdot}^2-dist_{\cdot j}^2+dist_{\cdot\cdot}^2)
$$

其中，$dist_{ij}=D_{ij}$。由此，可以根据降维前的距离矩阵$\pmb{D}$求得降维后距离不变的矩阵$\pmb{B}$。令$\pmb{C}_n=\pmb{I}_n-\frac1n\pmb{11}^\top$为中心化矩阵(Centering matrix)，则，

$$
\pmb{B}=-\frac12\pmb{C}_n\pmb{D}\pmb{C}_n
$$

其中，$\pmb{X}\pmb{C}_n$相当于对$\pmb{X}$的所有行向量减去行向量均值；$\pmb{C}_n\pmb{X}$相当于对$\pmb{X}$的所有列向量减去列向量均值；

- 获得降维后的样本投影矩阵$\pmb{Z}$

&emsp;&emsp;对矩阵$\pmb{B}$（实对称矩阵）做特征值分解，$\pmb{B}=\pmb{V\Lambda V}^\top$。假设有$d_*$个非零特征值构成对角矩阵$\pmb{\Lambda}_*=\textrm{diag}(\lambda_1,\lambda_2,...,\lambda_{d_*})$,以及所对应的特征向量矩阵$\pmb{V}_*$，则$\pmb{Z}$可以表示为，

$$
\pmb{Z}=\pmb{\Lambda}_*^{\frac{1}{2}}\pmb{V}_*^\top \in \mathbb{R}^{m\times d_*}
$$

&emsp;&emsp;现实应用中，可以选择$d'<d$个最大特征值构成的对角阵$\hat{\pmb{\Lambda}}$及特征向量矩阵$\hat{\pmb{V}}$，即

$$
\pmb{Z}=\hat{\pmb{\Lambda}}^{\frac{1}{2}} \hat{\pmb{V}}^\top \in \mathbb{R}^{m\times d'}
$$

### 算法

&emsp;&emsp;**输入**：距离矩阵$\pmb{D}$，低维空间维数$d'$.

&emsp;&emsp;**过程**：

&emsp;&emsp;&emsp;&emsp;1. 计算$\pmb{D}$;

&emsp;&emsp;&emsp;&emsp;2. 计算矩阵$\pmb{B}$;

&emsp;&emsp;&emsp;&emsp;3. 矩阵$\pmb{B}$做特征值分解；

&emsp;&emsp;&emsp;&emsp;4. 选取$\hat{\pmb{V}},\hat{\pmb{\Lambda}}$；

&emsp;&emsp;**输出**： 矩阵$\hat{\pmb{V}}\hat{\pmb{\Lambda}}^{1/2}$每一行即为一个样本的低维坐标。

- 示例代码

```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA   # 与MDS进行对比
from sklearn.manifold import MDS
    
ris = datasets.load_iris()
X = iris.data
y = iris.target

plt.subplot(121)
pca = PCA(n_components=2)
pca.fit(X)
new_X_pca = pca.transform(X)
plt.scatter(new_X_pca [:,0], new_X_pca [:,1], c=y)

plt.subplot(122)
mds = MDS( n_components=2, metric=True)
new_X_mds = mds.fit_transform(X)
plt.scatter(new_X_mds[:,0], new_X_mds[:,1], c=y)

plt.show()
```

## 等度量映射

&emsp;&emsp;等度量映射(Isometric Mapping, Isomap)的基本出发点在于，Isomap认为低维流行嵌入到高维空间之后，直接在高维空间计算直线距离具有误导性，因为高维空间的直线距离在低维流行是不可达的（如：瑞士卷上两个点（位于同一$x,y$坐标，$z$不同坐标）是不能用直线距离来计算的，因为该流行是扭曲过的）。



&emsp;&emsp;所谓$d$维流形是$n$维空间$(d<n)$的一部分，局部类似于$d$维超平面。例如：2D流形是一个2D形状，该形状可以在更高维的空间中弯曲和扭曲。**流形学习**通过训练实例所在的流形进行建模。流形学习基于流行假设，即大多数现实世界的高维数据集都接近于低维流形。如：三维空间的球面，其实可以只用经度和纬度两个特征来表示。低维嵌入流形上的本真距离（即测地线距离，如：北京至上海的距离（地球是圆的，直线距离要穿过地下层））不能用高维空间的直线距离来计算，但能用近邻距离来近似。


&emsp;&emsp;如何计算测地线距离呢？利用流形在局部与欧氏空间同胚这个性质，对每个样本点基于欧氏距离找出其近邻点，建立一个近邻连接图。于是，计算两点之间的测地线距离的问题就转变为计算近邻连接图上两点之间最短路径的问题。近邻图计算两点之间的最短路径，可以采用Dijkstra算法或Floyd算法，在得到任意两点的距离之后，就可以用多维缩放(MDS)方法来获得样本点在低维空间的坐标。

&emsp;&emsp;它的**核心思想**是沿着图的边移动的距离近似于沿着流形移动的距离。

### 算法

**输入**：样本集$\mathcal{D}=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$，低维空间维数$d'$.

**过程**：

1. 确定每个样本$\pmb{x}_i$的$k$近邻;

2. 使用最短路径算法(例如：Dijkstra)计算$k$近邻图的任意样本间距离$dist(\pmb{x}_i,\pmb{x}_j)$;

$$
 dist(\pmb{x}_i,\pmb{x}_j)=\left\{\begin{array}{ll} dist(\pmb{x}_i,\pmb{x}_j), & \pmb{x}_j\textrm{ is a nearest neighbor of }\pmb{x}_i\\ \infty, & \textrm{otherwise.}\end{array}\right.
$$

3. 以$dist(\pmb{x}_i,\pmb{x}_j)$为输入，使用MDS计算低维坐标；

**输出**： MDS计算的低维坐标。

- 示例

```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import Isomap

iris=datasets.load_iris()
X=iris.data
y=iris.target

fig,ax=plt.subplots(1,3,figsize=(15,5))
for idx,neighbor in enumerate([2,20,100]):
    isomap=Isomap(n_components=2, n_neighbors=neighbor)
    X_new=isomap.fit_transform(X)
    ax[idx].scatter(X_new[:,0],X_new[:,1],c=y)
    ax[idx].set_title("Isomap(n_neighbors=%d)"%neighbor)
plt.show()


```

### 流形

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

## LLE局部线性嵌入

### LLE基本思想

&emsp;&emsp;Isomap试图**保持**局部近邻样本之间的**距离**。LLE则试图**保持**局部邻域内样本之间的**线性关系**。假设样本$\pmb{x}_i$的坐标可由邻居样本$\pmb{x}_j,\pmb{x}_k,\pmb{x}_l$的坐标通过线性组合重构出来，即，

$$
\pmb{x}_i=w_{ij}\pmb{x}_j+w_{ik}\pmb{x}_k+w_{il}\pmb{x}_l
$$

则LLE希望此关系在低维空间依旧能得以保持。

### LLE求解

&emsp;&emsp;**Step 1**. 寻找样本$\forall \pmb{x}_i\in \mathcal{X}$的$k$个近邻。


&emsp;&emsp;**Step 2**. 求解重构系数矩阵$\pmb{W}$。

$$
\begin{split}
\min_W \quad &\varepsilon(\pmb{W})= \sum_{i=1}^m \left\lVert \pmb{x}_i-\sum_{j\in \mathcal{N}_i} w_{ij}\pmb{x}_j \right\rVert^2\\
\text{s.t.}\quad &\sum_{j\in \mathcal{N}_i} w_{ij}=1
\end{split}
$$

令$\varepsilon_i=\left\lVert \pmb{x}_i-\sum_{j\in \mathcal{N}_i} w_{ij}\pmb{x}_j \right\rVert^2$，则有，

$$
\begin{split}
\varepsilon_i&=\left\lVert \pmb{x}_i-\sum_{j\in \mathcal{N}_i} w_{ij}\pmb{x}_j \right\rVert^2\\
&=\left\lVert (w_{i1}+w_{i2}+\cdots+w_{ik})\pmb{x}_i-\sum_{j\in \mathcal{N}_i} w_{ij}\pmb{x}_j \right\rVert^2\quad \textrm{(sum of weights equals to 1)}\\
&=\left\lVert \sum_{j\in \mathcal{N}_i} w_{ij}\pmb{x}_i-\sum_{j\in \mathcal{N}_i} w_{ij}\pmb{x}_j \right\rVert^2\\
&=\left\lVert \sum_{j\in \mathcal{N}_i} w_{ij}(\pmb{x}_i-\pmb{x}_j) \right\rVert^2
\end{split}
$$

令$C_{jk}=(\pmb{x}_i-\pmb{x}_j)^T(\pmb{x}_i-\pmb{x}_k)$，则$w_{ij}$有闭式解，

$$
w_{ij}=\frac{\sum_{k\in\mathcal{N}_i} C_{jk}^{-1} }{ \sum_{l,s\in\mathcal{N}_i} C_{ls}^{-1} }
$$

&emsp;&emsp;**Step 3**. 恢复低维空间坐标。LLE在低维空间保持$\pmb{W}$不变，于是$\pmb{x}_i$对应的低维空间坐标$\pmb{z}_i$可以通过下式求解获得，

$$
\min\limits_{\pmb{Z}}\quad \sum_{i=1}^m \left\lVert \pmb{z}_i-\sum_{j\in \mathcal{N}_i}w_{ij}\pmb{z}_j\right\rVert^2
$$

&emsp;&emsp;上述两优化问题目标同形，唯一区别在于前一个问题要确定$\pmb{W}$，而后一个需要确定$\pmb{x}_i$所对应的低维坐标$\pmb{z}_i$。

$$
\varepsilon(\pmb{Z})&=\sum_{i=1}^n \left\Vert \pmb{z}_i-\sum_j w_{ij}\pmb{z}_j \right\Vert^2\\
$$

令,

$$
\pmb{W}=\begin{bmatrix}-&\pmb{w}_1&-\\-&\pmb{w}_2&-\\ &\vdots &\\ -&\pmb{w}_n&- \end{bmatrix} \quad \pmb{Z}=\begin{bmatrix}|&|&\cdots &|\\\pmb{z}_1&\pmb{z}_2&\cdots&\pmb{z}_n\\ |&|&\cdots &|\\ \end{bmatrix}
$$

则有，

$$
\begin{split}
\varepsilon(\pmb{Z})&=\Vert \pmb{Z}^\top -\pmb{WZ}^\top\Vert_F^2\\
&=\Vert (\pmb{I}-\pmb{W})\pmb{Z}^\top\Vert_F^2\\
&=\textrm{tr}(\pmb{Z}\pmb{M}\pmb{Z}^\top)
\end{split}
$$

其中，$\pmb{M}=(\pmb{I}-\pmb{W})^\top(\pmb{I}-\pmb{W})$。则问题可重写为，

$$
\begin{split}
\min_\pmb{Z}\quad &\text{tr}(\pmb{ZMZ}^\top)\\
\text{s.t.}\quad &\pmb{ZZ}^\top=\pmb{I}
\end{split}
$$

该问题可以通过特征值分解求得$\pmb{M}$的最大$d'$个特征值对应的特征向量组成的矩阵即为$\pmb{Z}^\top$。


## 拉普拉斯特征映射

### 拉普拉斯算子

- **连续型**

&emsp;&emsp;**[定义]**- 假设函数$f$连续二阶可微，则拉普拉斯算子($\Delta f$)由下式给出，

$$
\Delta f\triangleq\sum_{i=1}^n\frac{\partial^2f}{\partial x_i^2}=\nabla\cdot\nabla f=\textrm{div}(\textrm{grad}f)
$$

其中，

$$
\begin{split}
\textrm{grad} f&=\nabla f=\left(\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},...,\frac{\partial f}{\partial x_n}\right)\\
\textrm{div}(\textrm{grad}f)&=\nabla\cdot\nabla f=\frac{\partial^2 f}{\partial x_1^2}+\frac{\partial^2 f}{\partial x_2^2}+...+\frac{\partial^2 f}{\partial x_n^2}
\end{split}
$$

- **离散型**

&emsp;&emsp;在离散情况下，我们仍然希望拉普拉斯算子将输入的函数映射为其它函数。只不过，这种情况下，函数将被定义在离散的域上（如图$G$的有限顶点集$V$）。因此，我们可以将离散拉普拉斯算子$(\Delta)\phi(v)$作用在函数 $\phi :V\rightarrow R$。而$\phi$是一个定义在图的顶点集上的一个函数。我们也使用有限差分作为导数的离散类比，因此，我们不是使用导数来比较连续域的局部区域，而是使用有限差分来比较离散图的局部邻域。

&emsp;&emsp;对于一个定义在图$G$的顶点集的函数$\phi :V\rightarrow R$，离散拉普拉斯算子定义为，

$$
(\Delta\phi)(v_i)=\sum_{v_j\in \mathcal{N}(v_i)}W_{ij}[\phi(v_j)-\phi(v_i)]
$$

其中，$W_{ij}$为连接$v_i$和$v_j$的边$e_{ij}$的权值。

&emsp;&emsp;与连续版本一样，当$\phi(v_i)$的值比其周围的邻居大时(极大值)，离散拉普拉斯值较小；当$\phi(v_i)$的值比其周围的邻居小时（极小值），离散拉普拉斯值较大。

- 例：图像Laplacian算子

&emsp;&emsp;图像是一种离散型的数据，图像上的Laplacian算子可以大致进行如下运算。

$$
\begin{split}
\frac{\partial^2 f(x,y)}{\partial x^2}&=f_x^{''}(x,y)\\
&\approx f_x^{'}(x,y)-f_x^{'}(x-1,y)\\
&\approx f(x+1,y)-f(x,y)-f(x,y)+f(x-1,y)\\
&=f(x+1,y)+f(x-1,y)-2f(x,y)
\end{split}
$$

同理，$\frac{\partial^2 f(x,y)}{\partial y^2}=f(x,y+1)+f(x,y-1)-2f(x,y)$。因此有Laplacian值，

$$
\Delta f(x,y)=f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
$$

&emsp;&emsp;可以得出**结论**：Laplacian算子近似等于所有方向（自由度）差分累积（增益）。

- 例：图拉普拉斯算子

&emsp;&emsp;图数据上的Laplacian算子又该如何应用呢？图由$N$个结点及其连接边权值$W$所构成。和图像Laplacian算子类似，图Laplacian算子可以近似等于所有方向($\mathcal{N}_i$个邻接结点)的差分累积。

&emsp;&emsp;对于任意结点$i$，可以通过映射$f: V\rightarrow R$得到值$f_i$。显然结点$i$的Laplacian就等于其所有邻接点的差分累积，即

$$
(\Delta f)_i = \sum_{j\in\mathcal{N}_i}W_{ij}(f_j-f_i)
$$

因为$j\notin\mathcal{N}_i,W_{ij}=0$，上式可继续简化，

$$
\begin{split}
(\Delta f)_i&=\sum_j W_{ij}f_j -\sum_j W_{ij}f_i\\
&=(Wf)_i-(Df)_i\\
&=[(W-D)f]_i
\end{split}
$$

若$F: \mathbb{R}^d \rightarrow \mathbb{R}^p$,则有，

$$
\begin{split}
\Delta \pmb{F}&=\sum_{ij}\Vert \pmb{f}_i-\pmb{f}_j\Vert^2W_{ij}\\
&=\sum_{ij}\pmb{f}_i^\top\pmb{f}_iW_{ij}-2\sum_{ij}\pmb{f}_i^\top W_{ij}\pmb{f}_j+\sum_{ij}\pmb{f}_j^\top\pmb{f}_jW_{ij}\\
&=2\left(\sum_i \pmb{f}_i^\top D_{ii}\pmb{f}_i)-\sum_{ij}\pmb{f}_i^\top W_{ij}\pmb{f}_j\right)\\
&=2\textrm{tr}\left(\pmb{Y}^\top\pmb{L}\pmb{Y} \right)
\end{split}
$$




### 拉普拉斯矩阵

&emsp;&emsp;离散Laplacian算子表示为一个矩阵时，映射函数$\phi$可以写为列向量，$\Delta \phi$则表示为Laplacian矩阵$\pmb{L}$与列向量的乘积，即

$$
\Delta\phi=\pmb{L}\times \phi
$$

Laplacian矩阵$\pmb{L}=\pmb{D}-\pmb{W}$。其中，

$$
\pmb{D}_{ii}=\sum_j \pmb{W}_{ij}
$$



### 拉普拉斯变换

&emsp;&emsp;如果数据样本$i$与$j$很相似，则在拉普拉斯变换后的子空间与原空间一样，尽可能的接近。即，

$$
\begin{split}
\min\limits_{\pmb{Y}}\quad &\textrm{tr}(\pmb{Y}^\top\pmb{L}\pmb{Y})\\ 
\textrm{s.t.}\quad &\pmb{Y}^\top\pmb{DY}=\pmb{I}
\end{split}
$$

其中，$\pmb{L}$为拉普拉斯矩阵，$\phi:\pmb{x}\rightarrow\pmb{y},\pmb{x}\in\mathbb{R}^d, \pmb{y}\in\mathbb{R}^p$。

&emsp;&emsp;使用Lagrangian乘子法，可得，Lagrangian函数

$$
f(\pmb{Y})=\textrm{tr}(\pmb{Y}^\top\pmb{L}\pmb{Y})+ \textrm{tr}[\pmb{\Lambda}(\pmb{Y}^\top\pmb{DY}-\pmb{I})]
$$

对其求偏导数，

$$
\begin{split}
\frac{\partial f}{\partial \pmb{Y}}&=2\pmb{LY}+\pmb{D}^\top\pmb{Y}\pmb{\Lambda}^\top+\pmb{DY\Lambda}\\
&=2\pmb{LY}+2\pmb{DY\Lambda}
\end{split}
$$

令$\frac{\partial f}{\partial \pmb{Y}}=0$，可得$\pmb{LY}=-\pmb{DY\Lambda}$，即

$$
\pmb{Ly}=\lambda\pmb{Dy}
$$

&emsp;&emsp;因此，只需选择$p$个最小的特征值所对应的特征向量即可得到最优解$\hat{\pmb{Y}}$。
