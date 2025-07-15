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

## 去噪自编码器(Denoising AutoEncoder)

&emsp;&emsp;去噪自编码器（Denoising AutoEncoder）是一种自编码器的变体，其目标是在输入数据中添加噪声，然后训练模型从噪声中恢复原始数据。这样可以使模型学习到更鲁棒的特征表示。去噪自编码器的训练过程通常包括以下步骤：
1. 对输入数据添加噪声，生成噪声数据。
2. 使用噪声数据作为输入，原始数据作为目标，训练自编码器模型。
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


## 变分自编码器(Variational AutoEncoder)