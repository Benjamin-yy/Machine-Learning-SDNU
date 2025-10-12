#202311000207-冯小一

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 使用pandas从本地读取iris数据集
column_names = ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '品种']
iris_df = pd.read_csv('iris.data', header=None, names=column_names)
# 使用scikit-learn加载iris数据集
iris = load_iris()
X_sklearn, y_sklearn = iris.data, iris.target

# 分割本地数据的特征与标签
X_df = iris_df.iloc[:, :-1]  
y_df = iris_df.iloc[:, -1]   

# 将本地文件的文本标签转换为数值类型
le = LabelEncoder()
y_df_encoded = le.fit_transform(y_df)

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化逻辑回归分类器
model = LogisticRegression(max_iter=200)

# 本地文件数据集上进行训练和评估
y_pred_local = cross_val_predict(model, X_df, y_df_encoded, cv=kf)
accuracy_local = accuracy_score(y_df_encoded, y_pred_local)
precision_local = precision_score(y_df_encoded, y_pred_local, average='macro')
recall_local = recall_score(y_df_encoded, y_pred_local, average='macro')
f1_local = f1_score(y_df_encoded, y_pred_local, average='macro')

print("\n--- 使用pandas从本地读取iris数据集结果 ---")
print(f"准确率：{accuracy_local}")
print(f"精度：{precision_local}")
print(f"召回率：{recall_local}")
print(f"F1值：{f1_local}")

# 使用scikit-learn加载的数据集进行交叉验证
y_pred_sklearn = cross_val_predict(model, X_sklearn, y_sklearn, cv=kf)
accuracy_sklearn = accuracy_score(y_sklearn, y_pred_sklearn)
precision_sklearn = precision_score(y_sklearn, y_pred_sklearn, average='macro')
recall_sklearn = recall_score(y_sklearn, y_pred_sklearn, average='macro')
f1_sklearn = f1_score(y_sklearn, y_pred_sklearn, average='macro')

print("--- 使用scikit-learn加载iris数据集结果 ---")
print(f"准确率：{accuracy_sklearn}")
print(f"精度：{precision_sklearn}")
print(f"召回率：{recall_sklearn}")
print(f"F1值：{f1_sklearn}\n")