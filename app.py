import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# Define the SimpleRNN model class
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, question):
        embedded_question = self.embedding(question)
        hidden, final = self.rnn(embedded_question)
        output = self.fc(final.squeeze(0))
        return output

# Define tokenize function
def tokenize(text):
    text = text.lower()
    text = text.replace('?', '')
    text = text.replace("'", "")
    return text.split()

# Define text_to_indices function
def text_to_indices(text, vocab):
    indexed_text = []
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab['<UNK>'])
    return indexed_text

# Define predict function
def predict(model, question, vocab, threshold=0.5):
    numerical_question = text_to_indices(question, vocab)
    question_tensor = torch.tensor(numerical_question).unsqueeze(0)
    output = model(question_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    value, index = torch.max(probs, dim=1)
    if value < threshold:
        return "I don't know"
    return list(vocab.keys())[index]

# Initialize Flask app
app = Flask(__name__)

# Load the model and vocabulary
model = SimpleRNN(324)  # Assuming vocab size is 324
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Define the vocabulary
vocab = {'<UNK>': 0, 'what': 1, 'is': 2, 'the': 3, 'capital': 4, 'of': 5, 'france': 6, 'paris': 7, 'germany': 8, 'berlin': 9, 'who': 10, 'wrote': 11, 'to': 12, 'kill': 13, 'a': 14, 'mockingbird': 15, 'harper-lee': 16, 'largest': 17, 'planet': 18, 'in': 19, 'our': 20, 'solar': 21, 'system': 22, 'jupiter': 23, 'boiling': 24, 'point': 25, 'water': 26, 'celsius': 27, '100': 28, 'painted': 29, 'mona': 30, 'lisa': 31, 'leonardo-da-vinci': 32, 'square': 33, 'root': 34, '64': 35, '8': 36, 'chemical': 37, 'symbol': 38, 'for': 39, 'gold': 40, 'au': 41, 'which': 42, 'year': 43, 'did': 44, 'world': 45, 'war': 46, 'ii': 47, 'end': 48, '1945': 49, 'longest': 50, 'river': 51, 'nile': 52, 'japan': 53, 'tokyo': 54, 'developed': 55, 'theory': 56, 'relativity': 57, 'albert-einstein': 58, 'freezing': 59, 'fahrenheit': 60, '32': 61, 'known': 62, 'as': 63, 'red': 64, 'mars': 65, 'author': 66, '1984': 67, 'george-orwell': 68, 'currency': 69, 'united': 70, 'kingdom': 71, 'pound': 72, 'india': 73, 'delhi': 74, 'discovered': 75, 'gravity': 76, 'newton': 77, 'how': 78, 'many': 79, 'continents': 80, 'are': 81, 'there': 82, 'on': 83, 'earth': 84, '7': 85, 'gas': 86, 'do': 87, 'plants': 88, 'use': 89, 'photosynthesis': 90, 'co2': 91, 'smallest': 92, 'prime': 93, 'number': 94, '2': 95, 'invented': 96, 'telephone': 97, 'alexander-graham-bell': 98, 'australia': 99, 'canberra': 100, 'ocean': 101, 'pacific-ocean': 102, 'speed': 103, 'light': 104, 'vacuum': 105, '299,792,458m/s': 106, 'language': 107, 'spoken': 108, 'brazil': 109, 'portuguese': 110, 'penicillin': 111, 'alexander-fleming': 112, 'canada': 113, 'ottawa': 114, 'mammal': 115, 'whale': 116, 'element': 117, 'has': 118, 'atomic': 119, '1': 120, 'hydrogen': 121, 'tallest': 122, 'mountain': 123, 'everest': 124, 'city': 125, 'big': 126, 'apple': 127, 'newyork': 128, 'planets': 129, 'starry': 130, 'night': 131, 'vangogh': 132, 'formula': 133, 'h2o': 134, 'italy': 135, 'rome': 136, 'country': 137, 'famous': 138, 'sushi': 139, 'was': 140, 'first': 141, 'person': 142, 'step': 143, 'moon': 144, 'armstrong': 145, 'main': 146, 'ingredient': 147, 'guacamole': 148, 'avocado': 149, 'sides': 150, 'does': 151, 'hexagon': 152, 'have': 153, '6': 154, 'china': 155, 'yuan': 156, 'pride': 157, 'and': 158, 'prejudice': 159, 'jane-austen': 160, 'iron': 161, 'fe': 162, 'hardest': 163, 'natural': 164, 'substance': 165, 'diamond': 166, 'continent': 167, 'by': 168, 'area': 169, 'asia': 170, 'president': 171, 'states': 172, 'george-washington': 173, 'bird': 174, 'its': 175, 'ability': 176, 'mimic': 177, 'sounds': 178, 'parrot': 179, 'longest-running': 180, 'animated': 181, 'tv': 182, 'show': 183, 'simpsons': 184, 'vaticancity': 185, 'most': 186, 'moons': 187, 'saturn': 188, 'romeo': 189, 'juliet': 190, 'shakespeare': 191, 'earths': 192, 'atmosphere': 193, 'nitrogen': 194, 'bones': 195, 'adult': 196, 'human': 197, 'body': 198, '206': 199, 'metal': 200, 'liquid': 201, 'at': 202, 'room': 203, 'temperature': 204, 'mercury': 205, 'russia': 206, 'moscow': 207, 'electricity': 208, 'benjamin-franklin': 209, 'second-largest': 210, 'land': 211, 'color': 212, 'ripe': 213, 'banana': 214, 'yellow': 215, 'month': 216, '28': 217, 'days': 218, 'common': 219, 'february': 220, 'study': 221, 'living': 222, 'organisms': 223, 'called': 224, 'biology': 225, 'home': 226, 'great': 227, 'wall': 228, 'bees': 229, 'collect': 230, 'from': 231, 'flowers': 232, 'nectar': 233, 'opposite': 234, 'day': 235, 'south': 236, 'korea': 237, 'seoul': 238, 'bulb': 239, 'edison': 240, 'humans': 241, 'breathe': 242, 'survival': 243, 'oxygen': 244, '144': 245, '12': 246, 'pyramids': 247, 'giza': 248, 'egypt': 249, 'sea': 250, 'creature': 251, 'eight': 252, 'arms': 253, 'octopus': 254, 'holiday': 255, 'celebrated': 256, 'december': 257, '25': 258, 'christmas': 259, 'yen': 260, 'legs': 261, 'spider': 262, 'sport': 263, 'uses': 264, 'net,': 265, 'ball,': 266, 'hoop': 267, 'basketball': 268, 'kangaroos': 269, 'female': 270, 'minister': 271, 'uk': 272, 'margaretthatcher': 273, 'fastest': 274, 'animal': 275, 'cheetah': 276, 'periodic': 277, 'table': 278, 'spain': 279, 'madrid': 280, 'closest': 281, 'sun': 282, 'father': 283, 'computers': 284, 'charlesbabbage': 285, 'mexico': 286, 'mexicocity': 287, 'colors': 288, 'rainbow': 289, 'musical': 290, 'instrument': 291, 'black': 292, 'white': 293, 'keys': 294, 'piano': 295, 'americas': 296, '1492': 297, 'christophercolumbus': 298, 'disney': 299, 'character': 300, 'long': 301, 'nose': 302, 'grows': 303, 'it': 304, 'when': 305, 'lying': 306, 'pinocchio': 307, 'directed': 308, 'movie': 309, 'titanic': 310, 'jamescameron': 311, 'superhero': 312, 'also': 313, 'dark': 314, 'knight': 315, 'batman': 316, 'brasilia': 317, 'fruit': 318, 'king': 319, 'fruits': 320, 'mango': 321, 'eiffel': 322, 'tower': 323}

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    answer = predict(model, question, vocab)
    return jsonify({'answer': answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)