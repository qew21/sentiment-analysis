from flask import Flask, request, jsonify
from predict import load_model, predict_sentiment

app = Flask(__name__)
model, tokenizer, device = load_model()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    text = data['text']
    prediction = predict_sentiment(text, model, tokenizer, device)
    response = {'prediction': prediction}  # positive, negative, neutral
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
