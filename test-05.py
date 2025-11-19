# 202311000207-冯小一

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

iris = load_iris()
X = iris.data 
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=1/3, 
    random_state=42, 
    stratify=y
)

bp_model = MLPClassifier(
    hidden_layer_sizes=(10, 10),  
    activation='relu',           
    solver='adam',              
    max_iter=1000,               
    random_state=42             
)

bp_model.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
scores = {}

accuracy_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='accuracy')
precision_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='precision_weighted')
recall_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='recall_weighted')
f1_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='f1_weighted')

print("\n====================================================")
print("               202311000207-冯小一")
print("====================================================")

print("交叉验证性能")
print(f"准确度 (Accuracy)  : {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
print(f"精确率 (Precision) : {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
print(f"召回率 (Recall)    : {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
print(f"F1 值 (F1-score)   : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print("====================================================")

y_pred = bp_model.predict(X_test)

print("测试集性能")
print(f"准确度 (Accuracy) : {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"召回率 (Recall)   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 值 (F1-score)  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("====================================================")

print("分类报告")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("====================================================\n")