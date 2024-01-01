from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load the saved model architecture
json_file = open("model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Load the saved model weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text:
            input_sequence = tokenizer.texts_to_sequences([input_text])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            prediction = loaded_model.predict(padded_input_sequence)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            return render_template('index.html', input_text=input_text, predicted_label=predicted_label)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
