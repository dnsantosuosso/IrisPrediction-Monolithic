from sklearn import datasets
import pandas as pd

# Load iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Choose "petal length (cm)" as target, so no need to add the species target
# The resulting dataframe already contains "petal length (cm)" as one of its columns

# Save the dataframe to a CSV file
iris_df.to_csv('iris_data_for_linear_regression.csv', index=False)