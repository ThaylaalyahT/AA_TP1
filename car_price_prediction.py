import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


x_train = train[["milage"]]
y_train = train["price"]
x_test = test[["milage"]]
 
reg = LinearRegression()
reg.fit(x_train, y_train)
 
c = reg.intercept_
m = reg.coef_[0]
print(f"y = {m} * x + {c}")
 
plt.scatter(x_train, y_train, label="Dados Históricos", alpha=0.3)
plt.plot(x_test["milage"], m*x_test["milage"] + c, color='red', label="Regressão Linear")
plt.xlabel("Milage")
plt.ylabel("Price")
plt.legend()
plt.show()

