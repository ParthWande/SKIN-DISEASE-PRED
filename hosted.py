from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  
# Load and preprocess data
df = pd.read_csv('Training.csv')
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
df_selected = df[selected_columns].copy()
df_selected['prognosis'] = df_selected['prognosis'].apply(lambda x: x if x in skin_diseases else 'none')

# Prepare the data for model training
X = df_selected.drop('prognosis', axis=1)
y = df_selected['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Define prediction function
def predict_skin_disease(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)[0]
    return prediction

# Define the POST API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'patches_in_throat',
        'yellowish_skin', 'redness_of_eyes', 'dischromic _patches', 'puffy_face_and_eyes',
        'drying_and_tingling_lips', 'blister', 'red_sore_around_nose',
        'yellow_crust_ooze', 'pus_filled_pimples', 'blackheads', 'scurring',
        'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
        'inflammatory_nails'
    ]
    
    # Prepare the input data for prediction
    input_data = [data.get(symptom, 0) for symptom in symptoms]
    result = predict_skin_disease(input_data)
    
    return jsonify({'prognosis': result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
