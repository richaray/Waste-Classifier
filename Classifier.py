import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('C:/Users/RICHA/Downloads/recyclable_non_recyclable_dataset.csv')

# Extract object names and labels
object_names = df['Object Name'].values
labels = df['Label'].values

# OneHotEncode the object names
onehot_encoder = OneHotEncoder(sparse_output=False)
object_names_encoded = onehot_encoder.fit_transform(object_names.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(object_names_encoded, labels, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

def get_model_accuracy():
    """Return the accuracy of the model."""
    accuracy = model.score(X_test, y_test)
    return accuracy

def classify_object(detected_object: str):
    """Classify whether an object is recyclable or non-recyclable."""
    detected_object = detected_object.lower().strip()

    # Try to match the detected object with known object names
    matched_object = None
    for known_object in df['Object Name'].str.lower():
        if known_object in detected_object:
            matched_object = known_object
            break

    if matched_object:
        # Encode the matched object and classify it
        object_encoded = onehot_encoder.transform([[matched_object]])
        prediction = model.predict(object_encoded)

        return prediction[0], matched_object
    else:
        return None, None
