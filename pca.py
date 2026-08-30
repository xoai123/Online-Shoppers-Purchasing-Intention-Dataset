import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Đọc dữ liệu đã xử lý của bạn vào
# Giả sử file csv bạn vừa xuất tên là 'online_shoppers_processed.csv'
df_model = pd.read_csv('online_shoppers_processed.csv')

# Tách X (features) và y (target)
X = df_model.drop(columns=['Revenue'])
y = df_model['Revenue']

# 2. BẮT BUỘC: Chuẩn hóa dữ liệu trước khi làm PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. TỰ CODE PCA (MANUAL IMPLEMENTATION)
# ==========================================

# Bước A: Tính ma trận hiệp phương sai (Covariance Matrix)
# Vì X_scaled có các dòng là samples, các cột là features nên ta cần chuyển vị (.T)
covariance_matrix = np.cov(X_scaled.T)

# Bước B: Tính Trị riêng (Eigenvalues) và Vectơ riêng (Eigenvectors)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Bước C: Sắp xếp Trị riêng và Vectơ riêng giảm dần
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Bước D: Chọn số chiều (n_components = 2 để vẽ biểu đồ 2D)
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

# Bước E: Chiếu dữ liệu vào không gian mới (Tính Toạ độ PC1, PC2)
X_pca = np.dot(X_scaled, eigenvector_subset)

# Ép sang kiểu số thực thực sự để tránh dính số phức ảo (nếu có)
X_pca = np.real(X_pca) 

# Tạo DataFrame mới chứa kết quả PCA
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Revenue'] = y.values

# ==========================================
# 4. TRỰC QUAN HÓA (VISUALIZATION)
# ==========================================
sns.set_style('darkgrid')
plt.figure(figsize=(20, 8))

# Biểu đồ 1: Không gian 2D của PC1 và PC2
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Revenue', 
    data=df_pca, 
    palette='coolwarm', 
    alpha=0.6
)
plt.title('2D PCA of Online Shoppers Data (Manual Implementation)', fontsize=14)
plt.xlabel('First Principal Component (PC1)', fontsize=12)
plt.ylabel('Second Principal Component (PC2)', fontsize=12)

# Biểu đồ 2: Scree Plot (Xem lượng thông tin giữ lại của các PC đầu tiên)
plt.subplot(1, 2, 2)
exp_var = sorted_eigenvalues / np.sum(sorted_eigenvalues) # Tính tỷ lệ phương sai giải thích
cum_exp_var = np.cumsum(exp_var) # Tính tổng tích lũy

# Chỉ vẽ top 10 PC đầu tiên để nhìn cho rõ
plt.bar(range(1, 11), exp_var[:10], alpha=0.7, align='center', label='Individual Variance', color='steelblue')
plt.step(range(1, 11), cum_exp_var[:10], where='mid', label='Cumulative Variance', color='red', linewidth=2)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.xlabel('Principal Component Index', fontsize=12)
plt.title('Scree Plot (Top 10 Components)', fontsize=14)
plt.xticks(range(1, 11))
plt.legend(loc='best')

plt.tight_layout()
plt.show()