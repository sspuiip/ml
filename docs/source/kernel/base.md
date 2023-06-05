## 核函数基础

### 核函数的基本运算

1. 模长

$$
\Vert \phi(\pmb{x})\Vert=\sqrt{\langle \phi(\pmb{x}),\phi(\pmb{x})\rangle}=\sqrt{\kappa(\pmb{x},\pmb{x})}
$$

2. 标准化

$$
\hat{\phi}(\pmb{x})=\frac{\phi(\pmb{x})}{\Vert\phi(\pmb{x}) \Vert}
$$

$$
\hat{\kappa}(\pmb{x},\pmb{z})=\frac{\kappa(\pmb{x},\pmb{z})}{\sqrt{\kappa(\pmb{x},\pmb{x})\kappa(\pmb{z},\pmb{z})}}
$$

3. 线性组合S

$$
\left\Vert \sum_{i=1}^la_i\phi(\pmb{x}_i)\right\Vert^2=\sum_{i,j=1}^l a_ia_j\kappa(\pmb{x}_i\pmb{x}_j)
$$

4. 距离

$$
\left\Vert \phi(\pmb{x})-\phi(\pmb{z})\right\Vert^2=\kappa(\pmb{x},\pmb{x})-2\kappa(\pmb{x},\pmb{z})+\kappa(\pmb{z},\pmb{z})
$$

5. 均值

$$
\left\Vert \phi_S \right\Vert^2=\left\Vert \frac{1}{l}\sum_{i=1}^l\phi(\pmb{x}) \right\Vert^2=\frac1l\sum_{i=1}^l\sum_{j=1}^l\kappa(\pmb{x}_i,\pmb{x}_j)
$$

6. 样本均值距离

$$
\left\Vert \phi(\pmb{x})-\phi_S\right\Vert^2=\kappa(\pmb{x},\pmb{x})-\frac2l\sum_{i=1}^l\kappa(\pmb{x},\pmb{x}_i)+\frac{1}{l^2}\sum_{i=1}^l\sum_{j=1}^l\kappa(\pmb{x}_i,\pmb{x}_j)
$$

7. 平均样本均值距离

$$
\frac1l\sum_{i=1}^l\left\Vert \phi(\pmb{x}_i)-\phi_S\right\Vert^2=\frac1l\sum_{i=1}\kappa(\pmb{x}_i,\pmb{x}_j)-\frac{1}{l^2}\sum_{i=1}\sum_{j=1}\kappa(\pmb{x}_i,\pmb{x}_j)
$$

8. 中心化

- 样本中心化


$$
\tilde{\phi}(\pmb{x})=\phi(\pmb{x})-\phi_S
$$

- 核函数中心化

$$
\begin{split}
\tilde{\kappa}(\pmb{x},\pmb{z})&=\langle\phi(\pmb{x})-\phi_S,\phi(\pmb{x})-\phi_S\rangle\\
&=\kappa(\pmb{x},\pmb{z})-\frac1l\sum_{i=1}\kappa(\pmb{x}_i,\pmb{z})-\frac{1}{l}\kappa\sum_{i=1}\kappa(\pmb{x}_i,\pmb{x})+\frac{1}{l^2}\sum_{i,j=1}\kappa(\pmb{x}_i,\pmb{x}_j)\\
\tilde{\pmb{K}}&=\pmb{K}-\frac1l\pmb{K11}^\top-\frac1l\pmb{11}^\top\pmb{K}+\frac{1}{l^2}(\pmb{1}^\top\pmb{K}\pmb{1})\pmb{11}^\top
\end{split}
$$


### 投影

 $\phi(\pmb{x})$在向量$\pmb{w}$上的**投影**$P_{\pmb{w}}(\phi(\pmb{x}))$为，

$$
P_{\pmb{w}}(\phi(\pmb{x}))=\frac{\langle\phi(\pmb{x}),\pmb{w}\rangle}{\Vert\pmb{w}\Vert}\cdot\frac{\pmb{w}}{\Vert\pmb{w}\Vert}=\frac{\langle\phi(\pmb{x}),\pmb{w}\rangle}{\Vert\pmb{w}\Vert^2}\cdot\pmb{w}
$$

如果$\pmb{w}$已单位化，则有，

$$
P_{\pmb{w}}(\phi(\pmb{x}))=\langle\phi(\pmb{x}),\pmb{w}\rangle\cdot\pmb{w}=\pmb{w}\cdot\langle\phi(\pmb{x}),\pmb{w}\rangle=\pmb{w}\pmb{w}^\top\phi(\pmb{x})
$$

即，**正交投影**$ P_{\pmb{w}}^\bot \phi(\pmb{x}) $ 为，

$$
P_{\pmb{w}}^\bot\phi(\pmb{x})=(\pmb{I}-\pmb{ww}^\top)\phi(\pmb{x})
$$
