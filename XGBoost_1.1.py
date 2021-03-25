import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
start=time.clock()
# 载入数据集
dataset = loadtxt('G:\\2020summer\\Project\\XGboost\\1.csv', delimiter=",")
print(dataset[0][1])
print("Data insert successful!")
# split data into X and y
X = dataset[:, 0:2]
Y = dataset[:, 2]

# 把数据集拆分成训练集和测试集
seed = 3
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# 拟合XGBoost模型
model = XGBClassifier()
model.fit(X_train, y_train)

# 对测试集做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
end=time.clock()

print('Running time: %s Seconds'%(end-start))