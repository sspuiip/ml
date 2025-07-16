# 表示学习

## 自编码器(AutoEncoder)



&emsp;&emsp;自编码器（Autoencoder）是一种无监督学习的神经网络模型，用于学习输入数据的有效编码（表示）。它的基本目标是让输出尽可能地重构输入。自编码器通常包括二个部分：

1. 编码器（Encoder）：将原始输入数据压缩为一个低维的潜在表示（latent representation）。通常是一个神经网络。如：

$$
\pmb{h} = E_\phi(\pmb{x}) = \sigma(\pmb{Wx} + \pmb{b})
$$(encoder)

2. 解码器（Decoder）：将潜在表示解码回原始输入数据的近似值。通常也是一个神经网络。如：

$$
\pmb{\hat{x}} = D_\theta(\pmb{h}) = \sigma(\pmb{Gh} + \pmb{d})
$$(decoder)




&emsp;&emsp;对于任意$\pmb{x}\in\mathcal{X}$,

$$
\pmb{h}\triangleq E_\phi(\pmb{x})=\sigma(\pmb{Wx}+\pmb{b}),\quad \pmb{\hat{x}}\triangleq D_\theta (\pmb{h})=\sigma(\pmb{Gh}+\pmb{d})
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

:::{admonition} 示例代码
:class: dropdown

```python
# 1. 导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 2. 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=128,
    shuffle=False
)


# 3. 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # 将输出限制在 [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

#  4. 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5.训练模型
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
# 6.可视化重建结果
model.eval()
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        x_hat = model(x)
        break  # 只取一批

# 显示前8张原图与重建图
n = 8
plt.figure(figsize=(16, 4))
for i in range(n):
    # 原图
    plt.subplot(2, n, i+1)
    plt.imshow(x[i].cpu().view(28, 28), cmap='gray')
    plt.axis('off')
    
    # 重建图
    plt.subplot(2, n, i+n+1)
    plt.imshow(x_hat[i].cpu().view(28, 28), cmap='gray')
    plt.axis('off')

plt.suptitle("Top: Original | Bottom: Reconstructed", fontsize=16)
plt.show()
```
:::

## 稀疏自编码器(Sparse AutoEncoder)

&emsp;&emsp;稀疏自编码器（Sparse AutoEncoder）是一种自编码器的变体，其目标是在编码过程中强制大部分神经元保持非激活状态，从而学习到更稀疏的表示。稀疏性可以通过在损失函数中添加一个正则化项来实现。典型的稀疏自编码器的损失函数形式为：

$$
\mathcal{L}(\phi,\theta) = \mathbb{E}_{x\sim q}\left[d(\pmb{x},D_\theta (E_\phi (\pmb{x}))) + \lambda \cdot \text{Sparsity}(E_\phi (\pmb{x}))\right]
$$(sparse-ae-min)
其中，$\lambda$是稀疏性正则化的权重，$\text{Sparsity}$是一个衡量编码器输出稀疏性的函数。例如：

$$
\mathcal{L}(\phi,\theta) = \mathbb{E}_{x\sim q}\left[\Vert \pmb{x}-D_\theta (E_\phi (\pmb{x}))\Vert_2^2 + \lambda \cdot \sum_{j=1}^m \text{KL}(\rho || \hat{\rho}_j)\right]
$$(sparse-ae-min2)
其中，$\rho$是期望的稀疏激活率，$\hat{\rho}_j$是编码器第$j$个神经元的实际激活率。

:::{admonition} 示例代码
:class: dropdown

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def kl_divergence(rho, rho_hat):
    rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)  # 避免 log(0)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, rho=0.05, beta=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.rho = rho
        self.beta = beta

    def forward(self, x):
        z = torch.sigmoid(self.encoder(x))
        x_hat = torch.sigmoid(self.decoder(z))
        return x_hat, z

    def loss_function(self, x, x_hat, z):
        # 重建误差
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        # 稀疏性惩罚项
        rho_hat = torch.mean(z, dim=0)  # 平均激活值
        kl = kl_divergence(self.rho, rho_hat).sum()
        return recon_loss + self.beta * kl

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=256,
    shuffle=True
)


# 训练过程
model = SparseAutoencoder(input_size=784, hidden_size=64, rho=0.05, beta=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        x_hat, z = model(x)
        loss = model.loss_function(x, x_hat, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")


# 可视化隐藏层激活分布
model.eval()
sample_x, _ = next(iter(train_loader))
sample_x = sample_x.to(device)
_, z = model(sample_x)
z = z.detach().cpu().numpy()

plt.figure(figsize=(10, 3))
plt.bar(range(z.shape[1]), z[0])
plt.title("Hidden Layer Activations (1st sample)")
plt.xlabel("Hidden Unit Index")
plt.ylabel("Activation")
plt.show()
```
:::

## 去噪自编码器(Denoising AutoEncoder)

&emsp;&emsp;去噪自编码器（Denoising AutoEncoder）是一种自编码器的变体，其目标是在输入数据中添加噪声，然后训练模型从噪声中恢复原始数据。这样可以使模型学习到更鲁棒的特征表示。去噪自编码器的训练过程通常包括以下步骤：
1. 对输入数据添加噪声，生成噪声数据。
2. 使用噪声数据作为输入，原始数据作为目标，训练自编码器模型。

:::{admonition} 示例代码
:class: dropdown

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------
# 添加高斯噪声的变换
# ---------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.3):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

# ---------------------------
# 自定义 Dataset：输出 (noisy, clean)
# ---------------------------
class NoisyMNISTDataset(Dataset):
    def __init__(self, clean_dataset, noisy_dataset):
        self.clean = clean_dataset
        self.noisy = noisy_dataset

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean_img = self.clean[idx][0]   # Tensor
        noisy_img = self.noisy[idx][0]   # Tensor
        return noisy_img, clean_img

# ---------------------------
# 去噪自编码器模型
# ---------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ---------------------------
# 加载数据集
# ---------------------------
def get_dataloaders(batch_size=128):
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    transform_noisy = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
        AddGaussianNoise(0., 0.3)
    ])

    train_clean = datasets.MNIST('./data', train=True, download=True, transform=transform_clean)
    train_noisy = datasets.MNIST('./data', train=True, download=True, transform=transform_noisy)
    train_dataset = NoisyMNISTDataset(train_clean, train_noisy)

    test_clean = datasets.MNIST('./data', train=False, download=True, transform=transform_clean)
    test_noisy = datasets.MNIST('./data', train=False, download=True, transform=transform_noisy)
    test_dataset = NoisyMNISTDataset(test_clean, test_noisy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ---------------------------
# 训练模型
# ---------------------------
def train(model, train_loader, device, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for noisy_x, clean_x in train_loader:
            noisy_x, clean_x = noisy_x.to(device), clean_x.to(device)
            output = model(noisy_x)
            loss = criterion(output, clean_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# ---------------------------
# 可视化重建效果
# ---------------------------
def visualize(model, test_loader, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        noisy_x, clean_x = next(iter(test_loader))
        noisy_x, clean_x = noisy_x.to(device), clean_x.to(device)
        denoised_x = model(noisy_x)

    plt.figure(figsize=(18, 6))
    for i in range(num_samples):
        # noisy
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(noisy_x[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        # denoised
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(denoised_x[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        # clean
        plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
        plt.imshow(clean_x[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')

    plt.suptitle("Top: Noisy | Middle: Denoised | Bottom: Original", fontsize=16)
    plt.show()

# ---------------------------
# 主函数
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = DenoisingAutoencoder().to(device)

    train(model, train_loader, device, epochs=10)
    visualize(model, test_loader, device)

if __name__ == "__main__":
    main()
```
:::

## 变分自编码器(Variational AutoEncoder)

&emsp;&emsp;**变分自编码器**是一种带有概率建模思想的自编码器，相比普通自编码器，它不仅学习一个固定的隐空间向量，而是学习一个潜在变量的分布（通常是高斯分布）。VAE的**主要思想**是将输入编码为一个高斯分布（均值$\mu$，方差$\sigma^2$），然后从中采样一个$\pmb{z}$向量，再用解码器重构原始输入。

:::{figure-md}
![AutoEncoder](../img/vae-frame.png){width=500px}

变分自编码器示例
:::

| 特点    | 普通自编码器（AE） | 变分自编码器（VAE）           |
| ----- | ---------- | --------------------- |
| 编码器输出 | 一个确定向量 z   | 一个分布（均值 μ 和标准差 σ）     |
| 解码器输入 | 固定的 z      | 从分布中采样的 z \~ N(μ, σ²) |
| 表达形式  | 点表示        | 分布式表示（更适合生成）          |
| 生成能力  | 弱          | 强，可用于图像生成             |

&emsp;&emsp;变分自编码器本质上是一个有潜变量的概率模型，其**目标**是建模观测变量的生成分布$p(\pmb{x})$，即

$$
\pmb{x}\sim p_\theta(\pmb{x})=\int p_\theta(\pmb{x}|\pmb{z})p(\pmb{z})d\pmb{z}
$$(vae-px)
其中，$\pmb{x}$是观测变量；$\pmb{z}$是潜变量；$p(\pmb{z})$是潜变量的先验分布，通常取为标准正态分布，即$p(\pmb{z})=\mathcal{N}(\pmb{0},\pmb{I})$；$p_\theta(\pmb{x}|\pmb{z})$为解码器，通常是一个神经网络，用于从潜变量$\pmb{z}$生成观测变量$\pmb{x}$，$\theta$是解码器的参数。

&emsp;&emsp;**由于直接计算$p(\pmb{x})$通常是不可行的**，因为式{eq}`vae-px`中的积分是高维的，且潜变量$\pmb{z}$的分布$p_\theta(\pmb{z}|\pmb{x})$通常是未知的。为了克服这个问题，引入变分分布$q_\phi(\pmb{z}|\pmb{x})$来近似分布$p_\theta(\pmb{z}|\pmb{x})$，其中$\phi$是变分分布的参数，即

$$
q_\phi(\pmb{z}|\pmb{x})\approx p_\theta(\pmb{z}|\pmb{x})
$$(vae-qz)

这是变分推断的核心思想。使用Jensen不等式来构造一个变分下界（ELBO）来逼近真实的对数似然。

$$
\begin{split}
\log p_\theta(\pmb{x}) &= \log \int p_\theta(\pmb{x}|\pmb{z})p(\pmb{z})d\pmb{z} \\
&= \log \int q_\phi(\pmb{z}|\pmb{x})\frac{p_\theta(\pmb{x}|\pmb{z})p(\pmb{z})}{q_\phi(\pmb{z}|\pmb{x})}d\pmb{z} \\
&\geq \int q_\phi(\pmb{z}|\pmb{x})\log \frac{p_\theta(\pmb{x}|\pmb{z})p(\pmb{z})}{q_\phi(\pmb{z}|\pmb{x})}d\pmb{z} \\
&= \mathbb{E}_{q_\phi(\pmb{z}|\pmb{x})}\left[\log p_\theta(\pmb{x}|\pmb{z})\right] - D_{KL}(q_\phi(\pmb{z}|\pmb{x})||p(\pmb{z}))
\end{split}
$$(vae-elbo)

其中，$D_{KL}$是Kullback-Leibler散度，用于衡量两个分布之间的差异。上式中的第一项是重构误差，第二项是正则化项，用于控制潜变量的分布与先验分布的差异。由式{eq}`vae-elbo`的等式右边，可以得到一个**变分下界**，即

$$
\mathcal{L}(\theta, \phi;\pmb{x}) = \mathbb{E}_{q_\phi(\pmb{z}|\pmb{x})}\left[\log p_\theta(\pmb{x}|\pmb{z})\right] - D_{KL}(q_\phi(\pmb{z}|\pmb{x})||p(\pmb{z}))
$$(vae-elbo-def)  

其中，等式右边的$\mathbb{E}_{q_\phi(\pmb{z}|\pmb{x})}\left[\log p_\theta(\pmb{x}|\pmb{z})\right]$称为**证据下界**（Evidence Lower Bound, ELBO）。因为散度$D_{KL}(q_\phi(\pmb{z}|\pmb{x})||p(\pmb{z}))$是非负的，所以**最大化ELBO等价于最大化对数似然**$\log p(\pmb{x})$。

&emsp;&emsp;**一般来说，通常使用高斯假设来简化变分分布**$q_\phi(\pmb{z}|\pmb{x})$，即假设$q_\phi(\pmb{z}|\pmb{x})=\mathcal{N}(\mu_\phi(\pmb{x}),\sigma_\phi^2(\pmb{x}))$，其中$\mu_\phi(\pmb{x})$和$\sigma_\phi^2(\pmb{x})$是编码器的输出。先验分布$p(\pmb{z})$通常取为标准正态分布，即$p(\pmb{z})=\mathcal{N}(\pmb{0},\pmb{I})$。解码器输出$p_\theta(\pmb{x}|\pmb{z})$通常也是一个神经网络，用于从潜变量$\pmb{z}$生成观测变量$\pmb{x}$。解码器完成采样潜变量$\pmb{z}$(固定参数的正态分布采样)，再将$\pmb{z}$通过神经网络变换为$\pmb{x}$，其过程如下：

$$
\boxed{
\begin{split}
\pmb{z} \sim p(\pmb{z}) &= \mathcal{N}(\pmb{0},\pmb{I}) \\
\hat{\pmb{x}} &= \textrm{NeuroNet}(\pmb{z};\theta)\\
p(\pmb{x}|\pmb{z}) &= \mathcal{N}(\hat{\pmb{x}},\pmb{I}) \\
\end{split}}
$$(vae-decoder)

编码器则对应观测变量$\pmb{x}$到潜变量$\pmb{z}$的映射，其过程如下：

$$
\boxed{
\begin{split}
\pmb{\mu},\sigma  &= \textrm{NeuroNet}(\pmb{x};\phi)\\
q_\phi(\pmb{z}|\pmb{x}) &= \mathcal{N}(\pmb{z};\pmb{\mu},\sigma^2\pmb{I}) \\
\end{split} }
$$(vae-encoder)

&emsp;&emsp;需要注意的是，直接从分布中采样$\pmb{z}$会导致梯度无法传播，因此需要使用**重参数化技巧**（Reparameterization Trick）来解决这个问题。具体来说，可以将采样过程改为：

$$
\pmb{z} = \pmb{\mu} + \sigma \odot \pmb{\epsilon}
$$(vae-reparam) 

其中，$\pmb{\epsilon} \sim \mathcal{N}(\pmb{0},\pmb{I})$是一个标准正态分布的随机变量，$\odot$表示逐元素相乘。这样就可以将采样过程转化为一个确定性函数，从而使得梯度可以传播。

&emsp;&emsp;经过上述处理后，**变分自编码器的训练目标就变成了最大化ELBO，即最小化以下损失函数**：

$$
\mathcal{L}(\theta, \phi;\pmb{x}) = -\mathbb{E}_{q_\phi(\pmb{z}|\pmb{x})}\left[\log p_\theta(\pmb{x}|\pmb{z})\right] + D_{KL}(q_\phi(\pmb{z}|\pmb{x})||p(\pmb{z}))
$$(vae-loss)

其中，第一项是重构误差，第二项是KL散度。通常使用均方误差（MSE）作为重构误差的测度。将式{eq}`vae-decoder`和式{eq}`vae-encoder`代入式{eq}`vae-loss`，可以得到变分自编码器的最终损失函数：

$$
\boxed{
\begin{split}
\mathcal{L}(\theta, \phi;\pmb{x}) &= -\mathbb{E}_{\pmb{z}\sim q_\phi(\pmb{z}|\pmb{x})}\left[\log p(\pmb{x}|\pmb{z})\right] + \frac12\sum_{i=1}^D(\mu_i^2+\sigma_i^2-\log(\sigma_i^2)-1) \\
&= -\frac12\sum_{i=1}^D\left(x_i - \hat{x}_i\right)^2 + \frac12\sum_{i=1}^D(\mu_i^2+\sigma_i^2-\log(\sigma_i^2)-1) + \textrm{const} \\
\end{split}}
$$(vae-loss-final)

其中，期望项的计算使用了蒙特卡罗方法近似。确定损失函数后，就可以使用反向传播算法来优化模型参数$\theta$和$\phi$。    

:::{admonition} 示例代码
:class: dropdown

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# 参数
input_dim = 784  # 28x28 images
hidden_dim = 200
latent_dim = 20
batch_size = 32
epochs = 30
learning_rate = 3e-4

# x --> z
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma
    
# z --> x_hat
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat

def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction='sum')
        L2 = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return (L1 + L2) / batch_size
    
# 数据集
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(torch.flatten) # falatten
            ])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# 训练
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0
    for x, label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        cnt += 1
    loss_avg = loss_sum / cnt
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_avg:.4f}')
    losses.append(loss_avg)

# 可视化损失
epochs = list(range(1, epochs + 1))
plt.plot(epochs, losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 可视化重构结果
with torch.no_grad():
    sample_size = 64
    z = torch.randn(sample_size, latent_dim)
    x = model.decoder(z)
    generated_images = x.view(sample_size, 1, 28, 28)

grid_img = torchvision.utils.make_grid(generated_images, nrow=8, padding=2, normalize=True)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.show()
```
:::
