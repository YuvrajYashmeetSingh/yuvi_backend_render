from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle 
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}})

model = pickle.load(open('yuvraj_hate_speech(1).pkl', 'rb'))  # Loading ML model using pickle
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))  # Loading count vectorizer using pickle

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Preprocess function
def clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.replace('\n', ' ')  # Remove newlines
    text = ' '.join(word for word in text.split() if word not in stopwords_set)  # Remove stopwords
    text = ' '.join(stemmer.stem(word) for word in text.split())  # Stemming
    return text

@app.route('/', methods=['GET'])
def get_data():
    data = {
        "message": "API is Running 100"
    }
    return jsonify(data)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # Getting data from frontend
        print(data)
        preprocessed_text = clean(data)  # Preprocessing the data
        print(preprocessed_text)
        text_vectorized = cv.transform([preprocessed_text]).toarray()  # Extracting features
        prediction = model.predict(text_vectorized)  # Making prediction
        print(prediction)
        return jsonify({'Prediction': prediction[0]})  # Sending data to frontend
    except Exception as e:
        return jsonify({'error': str(e)})  # If there is error

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Process the file (assuming it's a text file)
    with open(filepath, 'r') as f:
        file_content = f.read()
    
    try:
        preprocessed_text = clean(file_content)  # Preprocessing the file content
        print(preprocessed_text)
        text_vectorized = cv.transform([preprocessed_text]).toarray()  # Extracting features
        prediction = model.predict(text_vectorized)  # Making prediction
        return jsonify({'Prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
