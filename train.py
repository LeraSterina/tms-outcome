import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv("tms_data.csv")

target = "outcome36"
numeric = ["score10","score20","score30","age"]
categorical = ["sex","education","site"]

X = df[numeric + categorical]
y = df[target]

pre = ColumnTransformer(
    [("num", StandardScaler(), numeric),
     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="drop",
)

model = Ridge(alpha=1.0)
pipe = Pipeline([("pre", pre), ("model", model)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)

print("R2:", round(r2_score(yte, pred), 3))
print("MAE:", round(mean_absolute_error(yte, pred), 3))

joblib.dump({"pipeline": pipe, "numeric": numeric, "categorical": categorical}, "model.pkl")
print("Saved model.pkl")
