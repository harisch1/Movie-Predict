from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score
import preprocess
import numpy as np

def Poly():
    X, y = preprocess.preProcess()
    y = np.reshape(y, (X.shape[0],1))

    alpha = 0
    deg = X.shape[1]

    model = Ridge(alpha=alpha, solver='auto', random_state = 42)  
    model = Pipeline([
            ("poly_features", PolynomialFeatures(degree=deg, include_bias=True)),
            ("std_scaler", StandardScaler()),
            ("regul_reg", model),
        ])
    model.fit(X, y)
    #y_pred = model.predict(x_test)
    cross_valid = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv = 5)
    print('Cross Validation Errors:', -np.mean(cross_valid))
    print('Theta: \n', model.named_steps["regul_reg"].coef_)

Poly()