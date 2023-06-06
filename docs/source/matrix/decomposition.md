# 矩阵分解

## QR分解

### QR的一般形式

&emsp;&emsp;对于一般矩阵$\pmb{A}_{m\times n}(m\ge n)$，可以分解为，

$$
\pmb{A}_{m\times n}=\pmb{Q}_{m\times m}\pmb{R}_{m\times n}=\pmb{Q}\left[ \begin{array}{c}\pmb{R}_1\\\pmb{0}\end{array}\right]=\left[ \pmb{Q}_1\quad \pmb{Q}_2\right]\left[ \begin{array}{c}\pmb{R}_1\\\pmb{0}\end{array}\right]=\pmb{Q}_1\pmb{R}_1
$$

其中$\pmb{Q}$为正交矩阵$\pmb{Q}^\top=\pmb{Q}^{-1}$（列向量正交），$\pmb{R}$为上三角矩阵，即$\pmb{R}_1$为$n\times n$的上三角矩阵。一般来说，$\pmb{Q}$矩阵的前$k(1\le k\le n)$个列向量构建了矩阵$\pmb{A}$的前$k$列张成的一个标准正交基。

- **例**1

```matlab
>>>A=[1 2 3;2 4 6;1 1 1]
>>>[Q,R]=qr(A)
>>>Q

   -0.4082   -0.1826   -0.8944
   -0.8165   -0.3651    0.4472
   -0.4082    0.9129   -0.0000
   
>>>R

   -2.4495   -4.4907   -6.5320
         0   -0.9129   -1.8257
         0         0    0.0000
         
>>>Q1=Q(:,1:2)

   -0.4082   -0.1826
   -0.8165   -0.3651
   -0.4082    0.9129
   
>>>R1=R(1:2,:)

   -2.4495   -4.4907   -6.5320
         0   -0.9129   -1.8257
         
>>>Q1*R1

    1.0000    2.0000    3.0000
    2.0000    4.0000    6.0000
    1.0000    1.0000    1.0000
```

- **注意**

&emsp;&emsp;当矩阵列不满秩时，可以使用`[Q,R,P]=qr(A)`来分解，且满足$\pmb{A}*\pmb{P}=\pmb{Q}*\pmb{R}$其中$\pmb{P}$为转换矩阵。若$\pmb{A}=[\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_n]$为数据矩阵时，$\pmb{P}$的作用等价于样本重新排列顺序。

```
>>>C=[1 1 2;2 1 4;3 1 6]

     1     1     2
     2     1     4
     3     1     6

>>>[Q,R,P]=qr(C)

Q =
   -0.2673    0.8729    0.4082
   -0.5345    0.2182   -0.8165
   -0.8018   -0.4364    0.4082

R =
   -7.4833   -1.6036   -3.7417
         0    0.6547    0.0000
         0         0    0.0000
         
P =
     0     0     1
     0     1     0
     1     0     0

>>>Q(:,1:rank(R)) * R(1:rank(R),:)
ans =

    2.0000    1.0000    1.0000
    4.0000    1.0000    2.0000
    6.0000    1.0000    3.0000

>>>C*P
ans =

     2     1     1
     4     1     2
     6     1     3
```




### QR分解的Gram-Schmidt正交化方法

&emsp;&emsp;考虑矩阵$\pmb{A}$，即，

$$
\pmb{A}=\left[\pmb{a}_1,\pmb{a}_2,...,\pmb{a}_n \right]
$$

Gram-Schmidt正交化过程如下，

$$
\begin{split}
\pmb{u}_1&=\pmb{a}_1,\quad \pmb{e}_1=\frac{\pmb{u}_1}{\Vert \pmb{u}_1\Vert}\\
\pmb{u}_2&=\pmb{a}_2-\langle\pmb{a}_2,\pmb{e}_1\rangle\pmb{e}_1,\quad\pmb{e}_2=\frac{\pmb{u}_2}{\Vert \pmb{u}_2\Vert}\\
&\vdots\\
\pmb{u}_{k+1}&=\pmb{a}_{k+1}-\sum_{i=1}^k\langle\pmb{a}_{k+1},\pmb{e}_i\rangle\pmb{e}_i\quad\quad\pmb{e}_{k+1}=\frac{\pmb{u}_{k+1}}{\Vert \pmb{u}_{k+1}\Vert}
\end{split}
$$

则QR分解结果为，

$$
\pmb{A}=\left[\pmb{a}_1,\pmb{a}_2,...,\pmb{a}_n \right]=\left[\pmb{e}_1,\pmb{e}_2,...,\pmb{e}_n \right]\begin{bmatrix}\langle\pmb{a}_1,\pmb{e}_1\rangle& \langle\pmb{a}_2,\pmb{e}_1\rangle &\cdots &\langle\pmb{a}_n,\pmb{e}_1\rangle\\ 0& \langle\pmb{a}_2,\pmb{e}_2\rangle &\cdots &\langle\pmb{a}_n,\pmb{e}_2\rangle\\ \vdots& \vdots& \ddots & \vdots &\\ 0& 0&\cdots&\langle\pmb{a}_n,\pmb{e}_n\rangle \end{bmatrix}=\pmb{QR}
$$

其中，$\langle \pmb{e}_i,\pmb{a}_i\rangle=\Vert\pmb{u}_i\Vert$。

&emsp;&emsp;注意，

$$
\langle \pmb{e}_i,\pmb{a}_i\rangle=\left\langle\frac{\pmb{u}_i}{\Vert\pmb{u}_i\Vert},\pmb{u}_i+\sum_{j=1}^{i-1}\langle\pmb{a}_i,\pmb{e}_j\rangle\pmb{e}_j\right\rangle=\left\langle\frac{\pmb{u}_i}{\Vert\pmb{u}_i\Vert},\pmb{u}_i\right\rangle=\Vert\pmb{u}_i\Vert
$$

以及，

$$
\begin{split}
\pmb{a}_i&=\sum_{j=1}^{i-1}\langle\pmb{a}_i,\pmb{e}_j\rangle\pmb{e}_j+\pmb{u}_i\\
&=\sum_{j=1}^{i-1}\langle\pmb{a}_i,\pmb{e}_j\rangle\pmb{e}_j+\frac{\pmb{u}_i}{\Vert\pmb{u}_i\Vert}\Vert\pmb{u}_i\Vert\\
&=\sum_{j=1}^{i-1}\langle\pmb{a}_i,\pmb{e}_j\rangle\pmb{e}_j+\pmb{e}_i\langle \pmb{e}_i,\pmb{a}_i\rangle\\
\end{split}
$$



## 特征值分解

### 方阵的特征值分解

&emsp;&emsp;若方阵$\pmb{A}_{m\times m}$可对角化，则有，

$$
\pmb{A}=\pmb{X}\pmb{\Lambda}\pmb{X}^{-1}
$$

&emsp;&emsp;矩阵$\pmb{A}$的特征向量是经过矩阵$\pmb{A}$变换后方向保持不变的向量，而特征值为这个变换中特征向量的缩放因子，即矩阵$\pmb{A}$对特征向量$\pmb{x}$的变换等于特征向量与特征值的乘积。

$$
\pmb{Ax}=\lambda \pmb{x}
$$

&emsp;&emsp;令，

$$
\pmb{X}=\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_n\\|&|&\dots&| \end{pmatrix},\quad \pmb{\Lambda}=\begin{pmatrix}\lambda_1&&&\\&\lambda_2&&\\&&\ddots&\\&&&\lambda_n \end{pmatrix}
$$

可以得到，

$$
\begin{split}\pmb{AX}&=A\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_n\\|&|&\dots&| \end{pmatrix}\\&=\begin{pmatrix}|&|&\dots&|\\\lambda_1\pmb{x}_1&\lambda_2\pmb{x}_2&\dots&\lambda_n\pmb{x}_n\\|&|&\dots&| \end{pmatrix}\\&=\begin{pmatrix}|&|&\dots&|\\\pmb{x}_1&\pmb{x}_2&\dots&\pmb{x}_n\\|&|&\dots&| \end{pmatrix}\begin{pmatrix}\lambda_1&&&\\&\lambda_2&&\\&&\ddots&\\&&&\lambda_n \end{pmatrix}\\&=\pmb{X\Lambda}\end{split}
$$

最终，我们可以得到结论（特征向量线性无关，故$\pmb{X}^{-1}$存在），

$$
\pmb{A}=\pmb{X}\pmb{\Lambda} \pmb{X}^{-1}
$$

#### 对称矩阵的特征值分解

&emsp;&emsp;任意对称实矩阵$\pmb{A}$可分解为，

$$
\pmb{A}=\pmb{Q}\pmb{\Lambda} \pmb{Q}^{-1}=\pmb{Q}\pmb{\Lambda} \pmb{Q}^\top.
$$

&emsp;&emsp;对称矩阵有一个**非常重要的性质**：
> 对称矩阵的特征向量是正交向量，即$\langle\pmb{x}_i, \pmb{x}_j\rangle=0,\forall i\neq j$。

&emsp;&emsp;**证**：假设$\lambda_1,\lambda_2,\pmb{x}_1,\pmb{x}_2$为对称矩阵$S$的任意互不相等的特征值和特征向量，那么，

$$
\lambda_1\langle\pmb{x}_i, \pmb{x}_j\rangle=\langle\pmb{Ax}_1, \pmb{x}_2\rangle=\pmb{x}_1^\top \pmb{Ax}_2=\pmb{x}_1^\top\lambda_2\pmb{x}_2=\lambda_2\langle\pmb{x}_1, \pmb{x}_2\rangle
$$

等式左边减去右边，得

$$
(\lambda_1-\lambda_2)\langle\pmb{x}_1, \pmb{x}_2\rangle=0.
$$

&emsp;&emsp;由于$\lambda_1\neq\lambda_2$，因此只能$\langle\pmb{x}_1, \pmb{x}_2\rangle=0$，即对称矩阵$\pmb{A}$的任意不相同的特征向量是正交的。于是，我们可以得到一个**重要结论**：
> 对称矩阵可以分解为两个由正交向量组成的矩阵与其对角阵的乘积，即$\pmb{A}=\pmb{Q}\pmb{\Lambda} \pmb{Q}^{-1}=\pmb{Q}\pmb{\Lambda} \pmb{Q}^\top$。



## 奇异值分解

&emsp;&emsp;对于$m\times n$的矩阵$\pmb{A}$，没有特征值的定义。因此，不能进行特征值分解。但可以使用奇异值分解，即

$$
\pmb{A}=\pmb{U}\pmb{\Sigma} \pmb{V}^\top=\sum_{i=1}\sigma_i\pmb{u}_i\pmb{v}_i^\top
$$

其中$\pmb{U}=(\pmb{u}_1,\pmb{u}_2,...,\pmb{u}_m)$为$m\times m$的正交矩阵，$\pmb{u}_i$称为矩阵$\pmb{A}$的左奇异向量。$\pmb{V}=(\pmb{v}_1,\pmb{v}_2,...,\pmb{v}_n)$为$n\times n$的正交矩阵，$\pmb{v}_i$称为矩阵$\pmb{A}$的右奇异向量。$\pmb{\Sigma}$的主对角线上的元素$\sigma_i$称为奇异值。

&emsp;&emsp;奇异值分解(SVD)的目标就是找到参数$\pmb{U},\pmb{V}$和奇异值矩阵$\pmb{\Sigma}$。$\pmb{u}_i,\pmb{v}_i$都是正交向量且满足，

$$
\pmb{Av}_i=\sigma_i\pmb{u}_i
$$

也就是说，一个$n$维向量$\pmb{v}$经过矩阵$\pmb{A}$的变换**等于**一个$m$维向量$\pmb{u}$经过奇异值$\sigma$的缩放。

### SVD参数求解

&emsp;&emsp;SVD参数一般通过$\pmb{AA}^\top$求$\pmb{U}$,$\pmb{A}^\top\pmb{A}$求$\pmb{V}$。

$$
\begin{split}
\pmb{AA}^\top&=\pmb{U}\pmb{\Sigma} \pmb{V}^\top \pmb{V\Sigma U}^\top=\pmb{U\Sigma}^2 \pmb{U}^\top\\ 
\pmb{A}^\top\pmb{A}&=\pmb{V}\pmb{\Sigma U}^\top \pmb{U\Sigma V}^\top=\pmb{V\Sigma}^2\pmb{V}^\top
\end{split}
$$

由于上述矩阵均为对称矩阵，因此，可以使用特征值分解求得矩阵$\pmb{U},\pmb{V}$。

&emsp;&emsp;$\pmb{\Sigma}$与$\pmb{AA}^\top,\pmb{A}^\top\pmb{A}$的关系，可以求得奇异值。

> $\pmb{AA}^\top$与$\pmb{A}^\top\pmb{A}$的特征值是相等的。

&emsp;&emsp;**证**：$\pmb{AA}^\top\pmb{x}=\lambda \pmb{x}$，可以得到$\pmb{A}^\top\pmb{AA}^\top\pmb{x}=\lambda \pmb{A}^\top\pmb{x}$，于是有，$\pmb{A}^\top\pmb{A}(\pmb{A}^\top\pmb{x})=\lambda(\pmb{A}^\top\pmb{x})$，即，$\lambda$为$\pmb{A}^\top\pmb{A}$的特征值，$\pmb{A}^\top\pmb{x}$为新的特征向量。

&emsp;&emsp;令$\lambda_i$为$\pmb{A}^\top\pmb{A}$的特征值，则，奇异值$\sigma_i=\sqrt{\lambda_i}$。