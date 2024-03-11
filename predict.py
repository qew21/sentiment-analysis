import torch
import transformers
from model import Transformer
import os
import pickle


def load_model():
    if os.path.exists("model/tokenizer.pkl"):
        tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained("./bert-base-uncased")
        pickle.dump(tokenizer, open("model/tokenizer.pkl", "wb"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists("model/model.pkl"):
        model = pickle.load(open("model/model.pkl", "rb"))
    else:
        model = Transformer("bert-base-uncased", 2, False)
        model = model.to(device)
        if not torch.cuda.is_available():
            state_dict = torch.load("model/transformer.pt", map_location=torch.device('cpu'))
        else:
            state_dict = torch.load("model/transformer.pt")
        model.load_state_dict(state_dict)
        pickle.dump(model, open("model/model.pkl", "wb"))
    return model, tokenizer, device


def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    result = "negative"
    if predicted_probability <= 0.8:
        result = "neutral"
    elif predicted_class == 1:
        result = "positive"
    print(f"{text} predicted result: {result}, predicted probability: {predicted_probability}")
    return  {'prediction': result, 'probability': predicted_probability}


if __name__ == "__main__":
    _model, _tokenizer, _device = load_model()
    for text in [
        "This film is terrible!",  # negative
        "This film is great!",  # positive
        "This film is not terrible!",  # positive
        "This film is not great!",  # negative
        "Where can I found this film?"  # neutral
    ]:
        predict_sentiment(text, _model, _tokenizer, _device)


