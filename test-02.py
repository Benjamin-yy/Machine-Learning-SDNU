# 202311000207-冯小一

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_cv = cross_val_score(nb_classifier, X, y, cv=cv, scoring='accuracy')
precision_cv = cross_val_score(nb_classifier, X, y, cv=cv, scoring='precision_macro')
recall_cv = cross_val_score(nb_classifier, X, y, cv=cv, scoring='recall_macro')
f1_cv = cross_val_score(nb_classifier, X, y, cv=cv, scoring='f1_macro')

print("\n"+"=" * 60)
print("                    202311000207-冯小一")
print("=" * 60)

print("交叉验证结果:")
print(f"Accuracy (Cross-validation): {np.mean(accuracy_cv):.4f} ± {np.std(accuracy_cv):.4f}")
print(f"Precision (Cross-validation): {np.mean(precision_cv):.4f} ± {np.std(precision_cv):.4f}")
print(f"Recall (Cross-validation): {np.mean(recall_cv):.4f} ± {np.std(recall_cv):.4f}")
print(f"F1 Score (Cross-validation): {np.mean(f1_cv):.4f} ± {np.std(f1_cv):.4f}")
print("=" * 60)

y_pred = nb_classifier.predict(X_test)

# 计算测试集上的性能评估指标
accuracy_test = accuracy_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred, average='macro')
recall_test = recall_score(y_test, y_pred, average='macro')
f1_test = f1_score(y_test, y_pred, average='macro')

# 输出测试集上的各项指标
print("Test Set Performance:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print("=" * 60)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# print(classification_report(y_test, y_pred))
print("=" * 60)