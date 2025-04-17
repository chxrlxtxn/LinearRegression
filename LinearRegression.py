from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


housing = fetch_openml(name="house_prices", as_frame=True)

print(housing.keys())
print(housing.DESCR)

df = pd.DataFrame(housing.data) # load numerical data
df.head(6)
print(df)
print(housing.feature_names)


# BASIC MANIPULATIONS ON THE DATA
numSamples = len(housing.target)
numFeatures = len(housing.feature_names)

print("num samples=", numSamples, "num features=", numFeatures)

y = np.array(housing.target)

ym = np.mean(y) # target mean
yp = 100* np.mean(y > 250000) # target percent above $250,000
print("The mean house price is ", "%.2f" % ym, "thousands of dollars.\nOnly", "%.1f" % yp, "percent are above $250k.")

# VISUALIZING THE DATA
x = np.array(df.get("1stFlrSF"))
plt.scatter(x,y)
plt.xlabel('Area (ft^2)')
plt.ylabel('Price ($)')
plt.grid(True)

# FITTING A SIMPLE LINEAR MODEL
def fit_linear(x,y):
    """
    Given vectors of data points (x,y), performs a fit for the linear model:
       yhat = beta0 + beta1*x, 
    The function returns beta0, beta1 and rsq, where rsq is the coefficient of determination.
    """
    # TODO complete the following code
    xm = np.mean(x) # feature mean
    ym = np.mean(y) # target mean

    sxx = np.mean((x-xm)**2)
    syy = np.mean((y-ym)**2)
    sxy = np.mean((x-xm)*(y-ym))
    beta1 = sxy/sxx
    beta0 = ym - beta1*xm
    rsq = sxy**2/sxx/syy

    return beta0, beta1, rsq

beta0, beta1, rsq = fit_linear(x,y)
print("B0:", "%.2f" % beta0, "B1:", "%.2f" % beta1, "rsq:", "%.2f" % rsq)

plt.plot(x, beta1*x + beta0, color='red')
plt.show()

# COMPUTE COEFFICIENTS OF DETERMINATION FOR ALL FEATURES
for str in df.columns:
    if df[str].dtype == np.int64:
        x = np.array(df.get(str))
        if (not np.isnan(x.all())):
            beta0, beta1, rsq = fit_linear(x,y)
            print('{}{}' .format(str.ljust(15), "%.3f" % rsq))
