import re
import string
from flask import Flask, request, jsonify
import pickle  
import nltk
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import render_template
import matplotlib.pyplot as plt
import io
import base64



lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nltk.data.path.append('./nltk_data')

app = Flask(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
  
unwanted_words = {'target', 'blank', 'http', 'www', 'src', 'img'}

def clean_text(text):
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|href\S+|@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = ' '.join([word for word in text.split() if word not in unwanted_words])
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words or word in {'not', 'no'}]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def create_chart(probabilities, classes):
    fig, ax = plt.subplots(figsize=(4,4))
    fig.patch.set_facecolor('none')  
    ax.set_facecolor('none') 
    ax.bar(classes, probabilities, color='skyblue') 
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')  
    ax.set_xlabel('Classes', color='white')
    ax.set_ylabel('Probability', color='white')
    ax.set_title('Class Probabilities', color='white')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)  #
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_str

def handle_negations(text):
    negations = {"not happy": "sad", "not sad": "happy", "not angry": "normal","not afraid": "surprised"}
    for phrase, replacement in negations.items():
        text = re.sub(phrase, replacement, text, flags=re.IGNORECASE)
    return text

#  the endpoint
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        processed_text = clean_text(text) 
        processed_text = handle_negations(processed_text)
        processed_text = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(processed_text)
        prediction = prediction[0]
        # probabilities = model.predict_proba(processed_text)[0]
        probabilities = model.predict_proba(processed_text)[0]
        class_labels = model.classes_
        
        # Create the chart
        chart = create_chart(probabilities, class_labels)

        return jsonify({'text': text, 'emotion': prediction, 'probabilities_chart': chart}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a basic route to check API status but now our home page tararata 
@app.route('/')
def home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
