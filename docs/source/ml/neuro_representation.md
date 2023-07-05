# 表示学习

## 自编码器(AutoEncoder)

&emsp;&emsp;自编码器是一种神经网络架构，目标是将输入信息压缩到指定的维度空间。一般由由以下两部分组成：

1. 原码空间$\mathcal{X}\in\mathbb{R}^n$与编码空间$\mathcal{Z}\in\mathbb{R}^m$。

2. 编码函数（参数为$\phi$）$E_\phi :\mathcal{X}\rightarrow\mathcal{Z}$与解码函数（参数为$\theta$）$D_\theta :\mathcal{Z}\rightarrow \mathcal{X}$。

<img alt="AutoEncdoer" src="../_images/Autoencoder.png" width='50%'/>

&emsp;&emsp;对于任意$\pmb{x}\in\mathcal{X}$,

$$
\pmb{z}\triangleq E_\phi(\pmb{x})=\sigma(\pmb{Wx}+\pmb{b}),\quad \pmb{\hat{x}}\triangleq D_\theta (\pmb{z})=\sigma(\pmb{Hz}+\pmb{d})
$$

其中，$\pmb{W},\pmb{H}$分别为编码层参数和解码层参数，$\sigma$为激活函数。

- **训练自编码器**

&emsp;&emsp;自编码器的目标是对于样本$\pmb{x}\sim q(\pmb{x}) \in\mathcal{X}$的重构误差测度$d:\pmb{X}\times\pmb{X}\rightarrow [0,\infty]$尽可能的小，即

$$
\min\limits_{\theta,\phi} \quad \mathcal{L}(\phi,\theta)=\mathbb{E}_{x\sim q}[d(\pmb{x},D_\theta (E_\phi (\pmb{x})))]
$$

&emsp;&emsp;一般来说，$q(\pmb{x})$取值为经验分布，

$$
q(\pmb{x})=\frac{1}{N}\sum_{i=1}^N \delta_{\pmb{x}_i}
$$

以及$d(\pmb{x},\pmb{x}')=\Vert \pmb{x}-\pmb{x}'\Vert_2^2$. 因此，寻找最估自编码器就等价于一个最小二乘优化，即

$$
\min\limits_{\theta,\phi}\quad\frac{1}{N}\sum_{i=1}^N \left\Vert \pmb{x}-D_\theta (E_\phi (\pmb{x}))\right\Vert_2^2
$$