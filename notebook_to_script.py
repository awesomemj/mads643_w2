import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    """
    Preprocess the input DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Convert 'datetime' column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract year, month, day, and weekend features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['weekend'] = (df['datetime'].dt.dayofweek == 5) | (df['datetime'].dt.dayofweek == 6)
    
    return df

def make_pipeline():
    """
    Create a data processing and modeling pipeline.
    Returns:
        Pipeline: Pipeline for data processing and modeling.
    """

    num_features = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                      ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features)])

    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])
    return pipe

def train(df):
    """
    Train the machine learning model using the input data.
    Args:
        df (dataframe): preprocessed dataframe.
    Returns:
        RandomForestRegressor: Trained machine learning model.
    """
    processed_df = preprocess_data(df)

    # Extract features and target
    y = processed_df['count']
    x = processed_df.drop('count', 1)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=1)

    # Build and train the model
    trained_model = make_pipeline()
    trained_model.fit(x_train, y_train)

    # Visualization
    figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
    figure.set_size_inches(18, 8)

    sns.barplot(data=df, x='year', y='count', ax=ax1)
    sns.barplot(data=df, x='month', y='count', ax=ax2)
    sns.barplot(data=df, x='day', y='count', ax=ax3)
    sns.barplot(data=df, x='hour', y='count', ax=ax4)
    sns.pointplot(data=df, x='hour', y='count', hue='dayofweek', ax=ax5)
    sns.barplot(data=df, x='dayofweek', y='count', ax=ax6)

    plt.show()

    # Store model metrics in a dictionary
    model_metrics = {
        "train_data": {
            "score": trained_model.score(x_train, y_train),
            "mae": mean_absolute_error(y_train, trained_model.predict(x_train)),
        },
        "test_data": {
            "score": trained_model.score(x_test, y_test),
            "mae": mean_absolute_error(y_test, trained_model.predict(x_test)),
        },
    }
    print(model_metrics)

    return trained_model

if __name__ == "__main__":
    import argparse
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="cleaned data file (CSV)")
    args = parser.parse_args()

    input_data = pd.read_csv(args.input_file)
    # Train the model
    model = train(input_data)