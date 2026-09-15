import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv("dataset.csv")
X=df.drop("price",axis=1); y=df["price"]
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),["location"]),
                       ("num","passthrough",["area_sqft","bedrooms","bathrooms","floors","parking"])])
model=Pipeline([("preprocessor",pre),("regressor",RandomForestRegressor(n_estimators=250,random_state=42,max_depth=12))])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("MAE:",mean_absolute_error(y_test,pred))
print("R2:",r2_score(y_test,pred))
joblib.dump(model,"model.pkl")
print("Model saved as model.pkl")
