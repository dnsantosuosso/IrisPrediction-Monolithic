import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import math

class MonolithicML:

    #Stage 1: Data is ingested from external sources
    def ingestion(self, data_path):
        data = pd.read_csv(data_path)
        return data

    def preparation(self, data):
        # Transformations as needed
        return data

    def combination(self, data_list):
        combined_data = pd.concat(data_list, axis=0)
        return combined_data

    def separation(self, data):
        #model will predict petal length based on petal width, sepal length and sepal width
        X = data.drop('petal length (cm)', axis=1)
        y = data['petal length (cm)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test


    def training(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluating(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    def serving(self, model, unlabeled_data):
        predictions = model.predict(unlabeled_data)
        return predictions

    def post_processing(self, predictions):
        processed_predictions = predictions # Simplified, can be enhanced
        return processed_predictions

    def monitoring(self, predictions):
        # For simplicity, we will just print them
        print(predictions)

    def pipeline(self, data_path):
        data = self.ingestion(data_path)
        prepared_data = self.preparation(data)
        combined_data = self.combination([prepared_data])
        X_train, X_test, y_train, y_test = self.separation(combined_data)
        model = self.training(X_train, y_train)
        mse = self.evaluating(model, X_test, y_test)
        print("Mean Squared Error: ", mse)
        print("On average, the model is ", math.sqrt(mse), "cm away from actual petal length values.")
        predictions = self.serving(model, X_test)
        post_processed_preds = self.post_processing(predictions)
        self.monitoring(post_processed_preds)

# Run the pipeline
ml = MonolithicML()
ml.pipeline('iris_data_for_linear_regression.csv')
