import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import seaborn as sns

df = pd.read_csv("creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']
df.info()

df.hist(figsize=(20,20))
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title('Box plot of Amount by Class')
plt.ylim([0, 300]) 
plt.show()

plt.figure(figsize=(10,8))
plt.plot(df['Time'], df['Amount'])
plt.title('Time vs Amount')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()

plt.figure(figsize=(15,10))
for column in df.columns[1:6]:
    sns.kdeplot(df[column], label=column)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x="V1", y="V2", hue="Class", data=df)
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mlflow.start_run()

# Modeli oluşturma
model = RandomForestClassifier()

# Modeli eğitme
model.fit(X_train, y_train)

y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auc_score = auc(recall, precision)
accuracy = model.score(X_test, y_test)
mlflow.log_metric('AUC', auc_score)
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "model")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auc_score = auc(recall, precision)
accuracy = model.score(X_test, y_test)

mlflow.log_metric('AUC', auc_score)
mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

!mlflow ui