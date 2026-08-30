import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ĐỌC DỮ LIỆU GỐC (TRƯỚC FEATURE ENGINEERING)
# ==========================================
# Đọc file dữ liệu thô ban đầu
df = pd.read_csv('online_shoppers_intention.csv')

# Sao chép dữ liệu để thực hiện mã hóa tối thiểu (bắt buộc để chạy được PCA)
df_encoded = df.copy()

# Chuyển đổi các cột dạng chuỗi (Object) và Boolean sang số bằng Label Encoding đơn giản
df_encoded['Month'] = pd.factorize(df_encoded['Month'])[0]
df_encoded['VisitorType'] = pd.factorize(df_encoded['VisitorType'])[0]
df_encoded['Weekend'] = df_encoded['Weekend'].astype(int)
df_encoded['Revenue'] = df_encoded['Revenue'].astype(int)

# Tách X (features - 17 đặc trưng gốc) và y (target - Revenue)
X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue']

# ==========================================
# 2. CHUẨN HÓA DỮ LIỆU (STANDARD SCALE)
# ==========================================
# PCA rất nhạy cảm với scale của dữ liệu, bắt buộc phải đưa về cùng scale (Mean=0, Var=1)
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

# Bước C: Sắp xếp Trị riêng và Vectơ riêng theo thứ tự giảm dần của Trị riêng
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Bước D: Chọn số chiều muốn giảm xuống (n_components = 2 để vẽ biểu đồ không gian 2D)
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

# Bước E: Chiếu dữ liệu từ không gian 17 chiều ban đầu xuống không gian mới 2 chiều (PC1, PC2)
X_pca = np.dot(X_scaled, eigenvector_subset)

# Loại bỏ phần ảo (nếu có do sai số tính toán số phức của np.linalg.eig)
X_pca = np.real(X_pca) 

# Tạo một DataFrame mới chứa tọa độ PC1, PC2 và nhãn Revenue để vẽ đồ thị
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Revenue'] = y.values

# Tính toán Tỷ lệ phương sai giải thích (Explained Variance Ratio) để làm Scree Plot
exp_var = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cum_exp_var = np.cumsum(exp_var)

print("--- KẾT QUẢ PHÂN TÍCH PCA TRƯỚC FEATURE ENGINEERING ---")
for i in range(min(5, len(exp_var))):
    print(f"PC{i+1}: Phương sai giải thích = {exp_var[i]*100:.2f}% (Tích lũy = {cum_exp_var[i]*100:.2f}%)")

# ==========================================
# 4. TRỰC QUAN HÓA KẾT QUẢ (VISUALIZATION)
# ==========================================
sns.set_style('darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Biểu đồ 1: Không gian 2 chiều của PC1 và PC2
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Revenue', 
    data=df_pca, 
    palette='coolwarm', 
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title('2D PCA of Online Shoppers Data\n(Before Feature Engineering - Manual Implementation)', fontsize=14)
axes[0].set_xlabel('First Principal Component (PC1)', fontsize=12)
axes[0].set_ylabel('Second Principal Component (PC2)', fontsize=12)

# Biểu đồ 2: Scree Plot cho toàn bộ 17 đặc trưng ban đầu
num_pcs = len(sorted_eigenvalues)
axes[1].bar(range(1, num_pcs + 1), exp_var, alpha=0.7, align='center', label='Individual Variance', color='steelblue')
axes[1].step(range(1, num_pcs + 1), cum_exp_var, where='mid', label='Cumulative Variance', color='red', linewidth=2)
axes[1].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[1].set_xlabel('Principal Component Index', fontsize=12)
axes[1].set_title('Scree Plot (All 17 Components)', fontsize=14)
axes[1].set_xticks(range(1, num_pcs + 1))
axes[1].legend(loc='best')

plt.tight_layout()
plt.savefig('pca_before_feature_engineering.png', dpi=300)
print("\nĐã lưu biểu đồ thành công vào file 'pca_before_feature_engineering.png'")
plt.show()