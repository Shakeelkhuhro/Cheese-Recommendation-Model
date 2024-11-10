# Cheese Recommendation and Prediction Model

This project is a **cheese recommendation and prediction model** that predicts whether a cheese is likely to be organic based on user preferences. It also provides recommendations for cheeses similar to those specified by the user.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Data](#data)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains a machine learning model built with **Python** and **Scikit-learn** that predicts if a cheese is organic and provides recommendations for similar cheeses based on user input. The model uses a combination of preprocessing, a supervised learning model, and dimensionality reduction techniques for similarity-based recommendations.

## Features

- **Organic Prediction**: Predict if a cheese, based on user preferences, is likely to be organic.
- **Cheese Recommendation**: Suggests similar cheeses that match user-defined characteristics.
- **User-Input Driven**: Allows users to input their preferences, making the model interactive and adaptable.

## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries: Install dependencies using the following command.

```bash
pip install pandas scikit-learn
```

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shakeelkhuhro/cheese-recommendation-model.git
    cd cheese-recommendation-model
    ```

2. Install the required packages if not already installed:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the `cheese_data.csv` file in the project root directory. This dataset should contain information on various cheese products, including features such as `ManufacturingType`, `CategoryType`, `FatLevel`, and `MoisturePercent`.

## Data

Ensure that the dataset (`cheese_data.csv`) is structured as follows:

- **Columns**:
  - `ManufacturingTypeEn`: Type of manufacturing (e.g., Farmstead, Industrial).
  - `CategoryTypeEn`: Category type (e.g., Firm Cheese, Semi-soft Cheese).
  - `FatLevel`: Fat level (e.g., lower fat).
  - `MoisturePercent`: Moisture percentage in the cheese.
  - `Organic`: Indicates if the cheese is organic (1 for organic, 0 for non-organic).

## Usage

### 1. Training the Model

The model pipeline is set up to preprocess data, train a `RandomForestClassifier`, and evaluate its performance. The model predicts if a cheese is likely to be organic and provides similarity-based recommendations.

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# Load and split data, preprocess, and train the model

```

### 2. Getting User Preferences

This function collects user inputs for cheese characteristics:

```python
def get_user_preferences():
    manufacturing_type = input("Manufacturing Type (e.g., Farmstead, Industrial): ")
    category_type = input("Category Type (e.g., Firm Cheese, Semi-soft Cheese): ")
    fat_level = input("Fat Level (e.g., lower fat): ")
    moisture_percent = float(input("Moisture Percent (e.g., 45.0): "))
    
    user_preferences = {
        'ManufacturingTypeEn': manufacturing_type,
        'CategoryTypeEn': category_type,
        'FatLevel': fat_level,
        'MoisturePercent': moisture_percent
    }
    return user_preferences
```

### 3. Making Predictions and Generating Recommendations

After collecting user preferences, predict if a cheese matching those preferences would likely be organic, and provide recommendations.

```python
user_preferences = get_user_preferences()
organic_prediction, probability = predict_user_cheese(user_preferences)
print(f"Prediction: The cheese is likely to be '{organic_prediction}' with probability {probability:.2f}")

recommended_cheeses = generate_recommendations(df, user_preferences)
print("Recommended Cheeses:")
print(recommended_cheeses)
```

### Example Output

```plaintext
Prediction: The cheese is likely to be 'Organic' with probability 0.85

Recommended Cheeses:
       CheeseName     CategoryTypeEn    FatLevel
0  Cheese A         Firm Cheese        lower fat
1  Cheese B         Semi-soft Cheese   regular fat
...
```

## Model Details

### Preprocessing

- **Numerical Features**: Imputed with the median and scaled.
- **Categorical Features**: Imputed with a constant value and one-hot encoded.
- **Pipeline**: Preprocessing is encapsulated in a `ColumnTransformer` and combined with the classifier in a single pipeline.

### Model

- **RandomForestClassifier**: A robust classifier for predicting whether a cheese is organic.
- **Evaluation Metrics**: Accuracy, Precision, Recall for classification performance.

### Recommendation System

- **PCA**: Used to reduce dimensionality for similarity calculations.
- **Euclidean Distance**: Measures similarity between user preferences and cheeses in the dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
