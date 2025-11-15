#202311000207-冯小一
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42, stratify=y)

clf = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=5, random_state=42)
clf.fit(X_train, y_train)
print("="*50)
print("                202311000207-冯小一")
print("="*50)
print("           在训练集上进行5折交叉验证评估")
print("="*50)

accuracy_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Accuracy scores: {np.round(accuracy_scores, 4)}")
print(f"Accuracy mean: {accuracy_scores.mean():.4f}\n")

precision_mean = cross_val_score(clf, X_train, y_train, cv=5, scoring='precision_weighted').mean()
recall_mean = cross_val_score(clf, X_train, y_train, cv=5, scoring='recall_weighted').mean()
f1_mean = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted').mean()

print(f"精度 :{precision_mean:.4f}")
print(f"召回率 :{recall_mean:.4f}")
print(f"F1值 :{f1_mean:.4f}\n")


print("="*50)
print("           在独立测试集上进行最终性能评估")
print("="*50)

y_pred = clf.predict(X_test)

print("【测试集分类报告】")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

test_accuracy = accuracy_score(y_test, y_pred)
print(f"【测试集总体准确率】: {test_accuracy:.4f}")
print("="*50)

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',   
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)

plt.title('测试集上的混淆矩阵', fontsize=16)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.show()