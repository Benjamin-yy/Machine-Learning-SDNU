# 202311000207-冯小一
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# 1. 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 使用留出法将数据集划分为训练集和测试集，留出 1/3 样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# 3. 使用训练集训练支持向量机 (SVM) 分类器 (SMO)
svm_model = SVC(kernel='linear', random_state=42)

svm_model.fit(X_train, y_train)

# 4. 使用五折交叉验证评估模型的性能
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = cross_val_score(svm_model, X, y, cv=cv, scoring='accuracy')
precision_scores = cross_val_score(svm_model, X, y, cv=cv, scoring='precision_weighted')
recall_scores = cross_val_score(svm_model, X, y, cv=cv, scoring='recall_weighted')
f1_scores = cross_val_score(svm_model, X, y, cv=cv, scoring='f1_weighted')

print("\n=========================================================")
print("                 202311000207-冯小一")
print("=========================================================")

print("交叉验证性能")
print(f"准确度 (Accuracy)  : {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
print(f"精确率 (Precision) : {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
print(f"召回率 (Recall)    : {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
print(f"F1 值 (F1-score)   : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print("=========================================================")

# 5. 使用测试集测试模型性能
y_pred = svm_model.predict(X_test)

print("测试集性能")
print(f"准确度 (Accuracy) : {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"召回率 (Recall)   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 值 (F1-score)  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("=========================================================")

# 6. 分类报告
print("分类报告")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("=========================================================\n")