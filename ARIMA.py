# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

import math
import pymysql
import warnings
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

connection = pymysql.connect(
    host="localhost",
    port=3306,
    user="username",
    passwd="password",
    db="test")

df = pd.read_sql_query("SELECT CONCAT(sheet, ':', featureA, ':', featureB) AS combination, CONCAT(year, '-', RIGHT(CONCAT('00', month), 2), '-01') AS date, value FROM fact_data_values ORDER BY combination, CAST(year AS unsigned), CAST(month AS unsigned)", con=connection)

combinations = list(df["combination"].unique())

p_values = [0, 1, 2, 4, 6]  # , 8, 10
d_values = [0, 1, 2]
q_values = [0, 1, 2]

scale_min = -100
scale_max = 100

train_size = 20

best_score = float("inf")
best_config = None

results = list()

warnings.filterwarnings("ignore")

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            tss = 0
            exceptions = 0

            for combination in combinations:
                subset = df.loc[df["combination"] == combination]
                ts = pd.Series(subset["value"].values, index=subset["date"])
                min, max = ts.min(), ts.max()

                try:
                    train = [(scale_min + (((scale_max - scale_min) * (x - min)) / (max - min))) for x in ts[0:train_size]]
                    test = [(scale_min + (((scale_max - scale_min) * (x - min)) / (max - min))) for x in ts[train_size:]]
                    predictions = list()

                    for t in range(len(test)):
                        model = ARIMA(train, order=order)
                        model_fit = model.fit(disp=0)
                        predictions.append(model_fit.forecast()[0])

                    mse = mean_squared_error(test, predictions)
                    tss += mse

                    output = str("{0};{1};{2};{3:0.4f};{4:0.4f};{5:0.4f};{6:0.4f};{7:0.4f};{8:0.4f};").format(order, combination, train, min, max, test[0], predictions[0][0], mse, tss)
                    results.append(output)
                    #print(output)

                except Exception as e:
                    #print("ERROR: %s" % str(e).split("\n")[0])
                    exceptions += 1
                    continue

            values_count = len(combinations) - exceptions
            avg_mse = math.sqrt(tss / values_count)
            if avg_mse < best_score:
                best_score, best_config = avg_mse, order

            output = str("ARIMA {0} ==> AVG-MSE {1:0.4f} out of {2} values").format(order, avg_mse, values_count)
            results.append(output)
            print(output)

output = str("BEST ARIMA {0} ==> AVG-MSE {1:0.4f}").format(best_config, best_score)
results.append(output)
print(output)

with open('results.csv', 'w') as file:
    for result in results:
        file.write(result + "\n")
