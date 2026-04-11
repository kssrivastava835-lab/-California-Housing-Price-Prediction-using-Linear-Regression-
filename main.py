from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.2)
pipe = Pipeline([
    ('scalar' , StandardScaler()),
    ('model', LinearRegression())
])

pipe.fit(X_train,y_train)
pred = pipe.predict(X_test)
print("MSE:",mean_squared_error(y_test,pred))