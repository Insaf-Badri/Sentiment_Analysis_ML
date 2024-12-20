import re
import string
from flask import Flask, request, jsonify
import pickle  
import nltk
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

nltk.data.path.append('./nltk_data')


app = Flask(__name__)

# Load the  model and the vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define a function to clean the text   
unwanted_words = {'target', 'blank', 'http', 'www', 'src', 'img'}
# Cleaning function
def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove URLs and mentions/hashtags in one step
    text = re.sub(r'http\S+|www\S+|href\S+|@\w+|#\w+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Replace emojis with their textual meaning
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove unwanted words
    text = ' '.join([word for word in text.split() if word not in unwanted_words])

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords but retain negations (like "not", "no")
    tokens = [word for word in tokens if word not in stop_words or word in {'not', 'no'}]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Rejoin tokens into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text


# Define the endpoint
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get the input text from the request
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        processed_text = clean_text(text) 
        processed_text = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(processed_text)
        predicted_emotion = prediction[0]

        return jsonify({'text': text, 'emotion': predicted_emotion})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a basic route to check API status
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Emotion Detection API is running!'})

if __name__ == '__main__':
    app.run(debug=True)
