import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Data transformation and loading


# Function that classify the tempo colum according to the normal tempo ranges
def tempoClassifier(x):
    if (x <= 40):
        x = 1  # Largo
    elif (x <= 66):
        x = 2  # Lento
    elif (x <= 76):
        x = 3  # Adagio
    elif (x <= 108):
        x = 4  # Andante
    elif (x <= 120):
        x = 5  # Moderato
    elif (x <= 168):
        x = 6  # Allegro
    else:
        x = 7  # Presto
    return x


dataMusic = pd.read_csv('./music_genre.csv')
dataMusic = dataMusic.dropna()

# Drop rows with duration values lees or equal to 0
dataMusic = dataMusic.drop(dataMusic[(dataMusic['duration_ms'] <= 0)].index)

# Drop rows with tempo values equals to ?
dataMusic = dataMusic.drop(dataMusic[(dataMusic['tempo'] == '?')].index)

# Change the type of tempo column
dataMusic['tempo'] = dataMusic['tempo'].astype('float64')

# Drop rows with tempo values lees or equal to 0
dataMusic = dataMusic.drop(dataMusic[(dataMusic['tempo'] <= 0)].index)

inlets = dataMusic[['acousticness', 'duration_ms',
                    'danceability', 'energy', 'instrumentalness', 'loudness', 'music_genre', 'speechiness', 'valence', 'tempo', 'liveness']]

# Diccionary to change the music_genere value from string to integer
genres = {'Electronic': 1, 'Anime': 2, 'Jazz': 3, 'Alternative': 4,
          'Country': 5, 'Rap': 6, 'Blues': 7, 'Rock': 8, 'Classical': 9, 'Hip-Hop': 10}
inlets['music_genre'] = inlets['music_genre'].map(genres)

# Classification of the tempo column
inlets['tempo'] = inlets['tempo'].apply(
    lambda x: tempoClassifier(x))

# Normalisation of the inlets data
transformer = MinMaxScaler()
inlets = transformer.fit_transform(inlets)
inlets = pd.DataFrame(inlets, columns=['acousticness', 'duration_ms',
                                       'danceability', 'energy', 'instrumentalness', 'loudness', 'music_genre', 'speechiness', 'valence', 'tempo', 'liveness'])

oulets = dataMusic['popularity']

x_train, x_test, y_train, y_test = train_test_split(
    inlets, oulets, test_size=0.2, random_state=42)

# Correlation Matrix
modelValues = inlets
modelValues['popularity'] = oulets
corr_df = modelValues.corr(method="pearson")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()

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
             0.005, f'{r2:.3f}', ha='center', color='black')

plt.title('Accuracy and Error of the models')
plt.xlabel('Model')
plt.ylabel('Value')
plt.legend('R_Squared')
plt.show()
