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
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['weekend'] = (df['datetime'].dt.dayofweek == 5) | (df['datetime'].dt.dayofweek == 6)
    return df

def make_pipeline():
    num_features = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                      ('scaler', StandardScaler())])

    cat_features = ['season', 'holiday', 'workingday', 'weather']
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))])

    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),
                                                   ('cat', cat_transformer, cat_features)])

    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])
    return pipe

def train(df):
    processed_df = preprocess_data(df)
    y = processed_df['count']
    x = processed_df.drop('count', 1)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=1)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="cleaned data file (CSV)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display metrics",
    )
    args = parser.parse_args()

    input_data = pd.read_csv(args.input_file)

    model = train(input_data)