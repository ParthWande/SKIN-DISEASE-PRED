import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
df = pd.read_csv('Training.csv')

# Define skin diseases and filter selected columns
skin_diseases = [
    'Fungal infection', 'Acne', 'Psoriasis', 'Allergy',
    'Drug Reaction', 'Chicken pox', 'Impetigo'
]

selected_columns = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'patches_in_throat',
    'yellowish_skin', 'redness_of_eyes', 'dischromic _patches', 'puffy_face_and_eyes',
    'drying_and_tingling_lips', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'prognosis'
]


# Select columns and filter for skin diseases, then make an explicit copy
df_selected = df[selected_columns].copy()
df_selected['prognosis'] = df_selected['prognosis'].apply(lambda x: x if x in skin_diseases else 'none')


# Save filtered data
df_selected.to_csv('skin_disease_data.csv', index=False)

# Split data into training and testing sets
X = df_selected.drop('prognosis', axis=1)
y = df_selected['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
#conf_matrix = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix:")
#print(conf_matrix)

# Prediction function
def predict_skin_disease(itching, skin_rash, nodal_skin_eruptions, patches_in_throat,
                         yellowish_skin, redness_of_eyes, dischromic_patches, puffy_face_and_eyes,
                         drying_and_tingling_lips, blister, red_sore_around_nose,
                         yellow_crust_ooze, pus_filled_pimples, blackheads, scurring,
                         skin_peeling, silver_like_dusting, small_dents_in_nails,
                         inflammatory_nails):

    input_data = pd.DataFrame({
        'itching': [itching],
        'skin_rash': [skin_rash],
        'nodal_skin_eruptions': [nodal_skin_eruptions],
        'patches_in_throat': [patches_in_throat],
        'yellowish_skin': [yellowish_skin],
        'redness_of_eyes': [redness_of_eyes],
        'dischromic _patches': [dischromic_patches],
        'puffy_face_and_eyes': [puffy_face_and_eyes],
        'drying_and_tingling_lips': [drying_and_tingling_lips],
        'blister': [blister],
        'red_sore_around_nose': [red_sore_around_nose],
        'yellow_crust_ooze': [yellow_crust_ooze],
        'pus_filled_pimples': [pus_filled_pimples],
        'blackheads': [blackheads],
        'scurring': [scurring],
        'skin_peeling': [skin_peeling],
        'silver_like_dusting': [silver_like_dusting],
        'small_dents_in_nails': [small_dents_in_nails],
        'inflammatory_nails': [inflammatory_nails]
    })

    prediction = model.predict(input_data)[0]
    return prediction

# Example prediction
prediction = predict_skin_disease(1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0)
print(f"Likely Prognosis: {prediction}")
