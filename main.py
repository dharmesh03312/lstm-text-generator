# ==============================
# 1. Imports
# ==============================
import numpy as np
import tensorflow as tf
import string
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# 2. Load & Clean Data
# ==============================
with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Remove Gutenberg header/footer (optional but better)
start = text.lower().find("start of")
end = text.lower().find("end of")
if start != -1 and end != -1:
    text = text[start:end]

# Lowercase + remove punctuation
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

print("Sample text:\n", text[:300])


# ==============================
# 3. Tokenization
# ==============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total words:", total_words)


# ==============================
# 4. Create Sequences
# ==============================
max_len = 20   # limit sequence size

input_sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        seq = token_list[max(0, i - max_len):i+1]
        input_sequences.append(seq)
        
print("Total sequences before limit:", len(input_sequences))


# ==============================
# 5. LIMIT DATA (IMPORTANT)
# ==============================
input_sequences = input_sequences[:50000]   # 🔥 reduce memory usage

print("Total sequences after limit:", len(input_sequences))


# ==============================
# 6. Padding
# ==============================
max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
)


# ==============================
# 7. Split X and y
# ==============================
X = input_sequences[:, :-1]
y = input_sequences[:, -1]   # ⚠️ NO one-hot encoding


# ==============================
# 8. Build Model
# ==============================
model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len - 1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',  # 🔥 memory-efficient
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# ==============================
# 9. Train Model
# ==============================
early_stop = EarlyStopping(monitor='loss', patience=3)

history = model.fit(
    X, y,
    epochs=30,           # keep small for speed
    batch_size=128,
    callbacks=[early_stop]
)


# ==============================
# 10. Helper Functions
# ==============================
index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def sample_with_temperature(preds, temperature=0.6):
    preds = np.asarray(preds).astype('float64')
    
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    
    preds = exp_preds / np.sum(exp_preds)   # ✅ FIXED LINE
    
    return np.random.choice(len(preds), p=preds)

def generate_text(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len - 1,
            padding='pre'
        )

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = sample_with_temperature(predicted[0])

        output_word = index_to_word.get(predicted_word_index, "")
        seed_text += " " + output_word

    return seed_text


# ==============================
# 11. Generate Text
# ==============================
while True:
    seed = input("\nEnter seed text (type 'exit' to quit): ")

    if seed.lower() == "exit":
        print("Exiting...")
        break

    try:
        num_words = int(input("How many words to generate: "))
    except ValueError:
        print("Please enter a valid number!")
        continue

    result = generate_text(seed, num_words)
    print("\nGenerated Text:\n", result)


# ==============================
# 12. Plot Loss
# ==============================
# plt.plot(history.history['loss'])
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()


# ==============================
# 13. Save Model
# ==============================
# model.save("lstm_text_model.h5")
# print("model saved")