import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Data transformation and loading
dataMusic = pd.read_csv('./music_genre.csv')
dataMusic = dataMusic.dropna()
dataMusic = dataMusic.drop(
    dataMusic[(dataMusic['duration_ms'] < 0)].index)
inlets = dataMusic[['acousticness', 'danceability',
                    'energy', 'instrumentalness', 'loudness']]
oulets = dataMusic['popularity']

x_train, x_test, y_train, y_test = train_test_split(
    inlets, oulets, test_size=0.2, random_state=42)

# Linear Regression model
linearModel = LinearRegression()
linearModel.fit(x_train, y_train)

# Random Forest Regressor model
randomForestModel = RandomForestRegressor(n_estimators=199, random_state=42)
randomForestModel.fit(x_train, y_train)

# Neural Network model with Scikit-Learn
neuralNetworkModel = MLPRegressor(hidden_layer_sizes=(128, 64), activation=(
    'relu'), max_iter=500, random_state=42, solver='adam', learning_rate_init=0.01)
neuralNetworkModel.fit(x_train, y_train)

# Predictions in all the models
yPredLinear = linearModel.predict(x_test)
yPredRandomForest = randomForestModel.predict(x_test)
yPredNeuralNetwork = neuralNetworkModel.predict(x_test)

# Calculation of the Coefficient of Determination (R Squared) for all the models
r2Linear = r2_score(y_test, yPredLinear)
r2RandomForest = r2_score(y_test, yPredRandomForest)
r2NeuralNetwork = r2_score(y_test, yPredNeuralNetwork)

# Create Bar Graphics in order to evaluate the accuracy of the models
models = ['Linear Regression', 'Random Forest', 'Neural Network']
r2Scores = [r2Linear, r2RandomForest, r2NeuralNetwork]

# R Squared and MSE Graphic
plt.figure(figsize=(10, 6))
bars = plt.bar(models, r2Scores, color='red', alpha=0.7)
for bar, r2 in zip(bars, r2Scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() +
             0.005, f'{r2:.2f}', ha='center', color='black')

plt.title('Accuracy and Error of the models')
plt.xlabel('Model')
plt.ylabel('Value')
plt.legend('R Squared')

plt.show()
