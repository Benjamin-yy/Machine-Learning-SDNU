# 202311000207-冯小一
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42, 
    stratify=y
)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

print("\n=========================================================")
print("                 202311000207-冯小一")
print("=========================================================")

print("交叉验证性能")

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

score_names = {
    'accuracy': '准确度 (Accuracy) ',
    'precision_macro': '精确率 (Precision)',
    'recall_macro': '召回率 (Recall)   ',
    'f1_macro': 'F1 值 (F1-score)  '
}

for score in scoring:
    cv_score = cross_val_score(kmeans, X_train, y_train, cv=5, scoring=score)
    name = score_names.get(score, score)
    print(f"{name}: {cv_score.mean():.4f} ± {cv_score.std():.4f}")

print("=========================================================")

y_pred = kmeans.predict(X_test)

print("测试集分类报告 (Classification Report)")
print(classification_report(y_test, y_pred))

print("=========================================================")

print("聚类中心点 (Cluster Centers)")
print(kmeans.cluster_centers_)
print("=========================================================\n")