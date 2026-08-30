# 🛒 Online Shoppers Purchasing Intention Prediction & Analysis

Dự án phân tích hành vi duyệt web và xây dựng mô hình học máy (Machine Learning) dự đoán khả năng hoàn tất đơn hàng (`Revenue`) của khách hàng trên nền tảng thương mại điện tử.

---

## 📌 Tổng quan đề tài
- **Mục tiêu:** Khám phá các yếu tố tác động đến quyết định mua sắm trực tuyến và huấn luyện mô hình **Random Forest Classifier** để phân loại phiên truy cập tiềm năng chuyển đổi doanh thu.
- **Bộ dữ liệu:** `Online Shoppers Purchasing Intention Dataset` gồm hơn 12.000 phiên giao dịch với 18 thuộc tính (thời gian xem trang, tỷ lệ thoát, giá trị trang, loại hệ điều hành, phân loại người dùng mới/cũ...).

---

## 🛠 Công nghệ & Thư viện sử dụng
- **Ngôn ngữ:** Python 3.x
- **Xử lý & Làm sạch dữ liệu:** `Pandas`, `NumPy`
- **Khám phá dữ liệu (EDA) & Trực quan hóa:** `Matplotlib`, `Seaborn`
- **Học máy (Machine Learning):** `Scikit-learn`
  - **Giảm chiều dữ liệu:** PCA (Principal Component Analysis)
  - **Mô hình phân loại:** Random Forest Classifier
  - **Đánh giá mô hình:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

---

## 📊 Cấu trúc dự án
```text
├── eda.ipynb                            # Quy trình khám phá dữ liệu, phân tích tương quan & huấn luyện Random Forest
├── online_shoppers_intention.csv        # Dữ liệu gốc
├── online_shoppers_processed.csv        # Dữ liệu sau khi làm sạch & mã hóa đặc trưng
├── pca.py / PCA1.py                     # Script thực thi giảm chiều dữ liệu PCA
└── pca_before_feature_engineering.png   # Biểu đồ không gian phân bố dữ liệu 2D sau PCA
