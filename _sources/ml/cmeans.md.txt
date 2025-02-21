# 聚类(一)


## KMeans


&emsp;&emsp;K-Means是一种流行的聚类算法，用于将数据集划分为$k$个聚类。它的工作原理是，基于最小化聚类内的方差，不断迭代地细化聚类分配。


### **1. 问题定义**


&emsp;&emsp;给定一个数据集 $ X = \{x_1, x_2, ..., x_N\} $，其中包含 $ d $ 维空间中的 $ N $ 个数据点，K-Means 的目标是将这些点划分为 $ k $ 个聚类 $ C_1, C_2, ..., C_k $，其中每个聚类由一个质心 $ \mu_j $ 表示。（Given a dataset $ X = \{x_1, x_2, ..., x_N\} $ with $ N $ data points in $ d $-dimensional space, the goal of K-Means is to partition these points into $ k $ clusters $ C_1, C_2, ..., C_k $, where each cluster is represented by a centroid $ \mu_j $.）

### **2. 目标函数**
&emsp;&emsp;该算法最小化每个点与其指定的聚类中心之间的平方距离（方差）的总和：

$$
J = \sum_{i=1}^{N} \sum_{j=1}^{k} \delta_{ij} \| x_i - \mu_j \|^2
$$(kmeans-obj-fun)

其中:
- $ \mu_j $ 是聚类$ j $的中心,
- $ \delta_{ij} $ 是一个指示变量，即，如果$ x_i $ 属于聚类$ j $则为1，否则为0，
- $ \| x_i - \mu_j \|^2 $ 为平方欧几里得距离。

目标函数确保数据点被分配到最近的聚类中心。


### **3. K-Means 算法步骤**
**步骤 1：初始化聚类中心**
- 从数据集中随机选择 $ k $ 个点作为初始聚类中心 $ \mu_1, \mu_2, ..., \mu_k $。
- 或者，使用**K-Means++**初始化来提高收敛性。

**步骤 2：将数据点分配给最近的聚类**

&emsp;&emsp;对于每个数据点 $x_i$，将其分配到具有最接近质心的聚类：

$$
C_j = \{x_i : \| x_i - \mu_j \|^2 \leq \| x_i - \mu_l \|^2, \forall l \neq j \}
$$(dist-from-centers)

**步骤 3：更新聚类中心**

&emsp;&emsp;将每个聚类的新质心计算为分配给它的所有点的平均值：

$$
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
$$(update-centers)

**步骤 4：检查收敛性**
- 如果聚类中心没有发生显著变化，则停止。
- 否则，重复**步骤 2**和**步骤 3**直到收敛。



### **4. 收敛性和复杂度**
**收敛性:**
- 当聚类分配不再改变或质心稳定时，K-Means 就会收敛。
- 它可能会卡在**局部最小值**，因此使用不同的初始化多次运行它会有所帮助。

**时间复杂度:**
- 将 $ N $ 个点分配给聚类：$ O(Nk) $
- 更新质心：$ O(N) $
- 迭代 $ IN $ 次：$ OF(INk) $
- **总体复杂度：**$ O(I Nk) $



### **5. K-Means++初始化（更好的初始化）**

&emsp;&emsp;与随机初始化不同，**K-Means++** 提高了收敛性：
1. 从数据中随机选择一个中心。
2. 选择后续中心，其概率与其与现有最近中心的平方距离成比例（Select subsequent centers with probability proportional to their squared distance from the closest existing center）。
3. 重复此操作直到选定 $ k $ 个中心。

&emsp;&emsp;该方法减少了由于初始化不良而导致聚类效果不佳的可能性。

---

### **6. 优点和缺点**
✅ **优点:**

- 简单快速。
- 适用于大型数据集。
- 当簇呈球形且分离良好时效果良好。

❌ **缺点:**

- 对初始质心敏感。
- 可能陷入局部最小值。
- 与非球形或重叠簇冲突。

---

### **7.Python实现**
以下是使用 `numpy` 实现的 K-Means 的简单实现：

```python
import numpy as np

def initialize_centroids(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])

def k_means(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
        
    return centroids, labels

# Example Usage
np.random.seed(42)
X = np.vstack((np.random.rand(50, 2) * 2, np.random.rand(50, 2) * 5 + 3))
centroids, labels = k_means(X, k=2)
print("Final Centroids:", centroids)
```

---

### **8. 与模糊 C 均值 (FCM) 的比较**

| 特征 | K-Means | Fuzzy C-Means (FCM) |
|:--:|:--:|:--:|
| 成员关系 | Hard (每个点属于一个簇) | Soft (每个点都有一定程度的属于多个聚类) |
| 收敛 | 快 | 慢 |
| 簇形状 | 适用于球形簇 | 适用于复杂的簇形状 |
| 敏感度 | 对初始质心敏感 | 更稳健但对参数敏感 |


## Fuzzy C-Means