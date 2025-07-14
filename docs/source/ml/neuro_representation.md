# 表示学习

## 自编码器(AutoEncoder)



&emsp;&emsp;自编码器（Autoencoder）是一种无监督学习的神经网络模型，用于学习输入数据的有效编码（表示）。它的基本目标是让输出尽可能地重构输入。自编码器通常包括二个部分：

1. 编码器（Encoder）：将原始输入数据压缩为一个低维的潜在表示（latent representation）。通常是一个神经网络。如：

$$
\pmb{z} = E_\phi(\pmb{x}) = \sigma(\pmb{Wx} + \pmb{b})
$$(encoder)

2. 解码器（Decoder）：将潜在表示解码回原始输入数据的近似值。通常也是一个神经网络。如：

$$
\pmb{\hat{x}} = D_\theta(\pmb{z}) = \sigma(\pmb{Hz} + \pmb{d})
$$(decoder)




&emsp;&emsp;对于任意$\pmb{x}\in\mathcal{X}$,

$$
\pmb{z}\triangleq E_\phi(\pmb{x})=\sigma(\pmb{Wx}+\pmb{b}),\quad \pmb{\hat{x}}\triangleq D_\theta (\pmb{z})=\sigma(\pmb{Hz}+\pmb{d})
$$(auto-encoder)

其中，$\pmb{W},\pmb{H}$分别为编码层参数和解码层参数，$\sigma$为激活函数。

:::{figure-md}
![AutoEncoder](../img/autoencoder.png){width=400px}

自编码器示例
:::

- **训练自编码器**

&emsp;&emsp;自编码器的目标是对于样本$\pmb{x}\sim q(\pmb{x}) \in\mathcal{X}$的重构误差测度$d:\pmb{X}\times\pmb{X}\rightarrow [0,\infty]$尽可能的小，即

$$
\min\limits_{\theta,\phi} \quad \mathcal{L}(\phi,\theta)=\mathbb{E}_{x\sim q}[d(\pmb{x},D_\theta (E_\phi (\pmb{x})))]
$$(ae-min0)

&emsp;&emsp;一般来说，$q(\pmb{x})$取值为经验分布，

$$
q(\pmb{x})=\frac{1}{N}\sum_{i=1}^N \delta_{\pmb{x}_i}
$$(ae-q)

以及$d(\pmb{x},\pmb{x}')=\Vert \pmb{x}-\pmb{x}'\Vert_2^2$. 因此，寻找最估自编码器就等价于一个最小二乘优化，即

$$
\min\limits_{\theta,\phi}\quad\frac{1}{N}\sum_{i=1}^N \left\Vert \pmb{x}-D_\theta (E_\phi (\pmb{x}))\right\Vert_2^2
$$(ae-min)


## 变分自编码器(Variational AutoEncoder)