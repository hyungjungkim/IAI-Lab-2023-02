# Source: https://victorzhou.com/blog/keras-rnn-tutorial/

from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

def prepareData(dir):
    data = text_dataset_from_directory(dir)
    return data.map(
        lambda text, label: (regex_replace(text, '<br />', ' '), label),
    )

train_data = prepareData('./train')
test_data = prepareData('./test')

for text_batch, label_batch in train_data.take(1):
    print(text_batch.numpy()[0])
    print(label_batch.numpy()[0]) # 0 = negative, 1 = positive

max_tokens = 1000
max_len = 100
vectorize_layer = TextVectorization(max_tokens=max_tokens,
                                    output_mode="int",
                                    output_sequence_length=max_len,
)

train_texts = train_data.map(lambda text, label: text)
vectorize_layer.adapt(train_texts)

model = Sequential()
model.add(Input(shape=(1,), dtype="string"))
model.add(vectorize_layer)
model.add(Embedding(max_tokens + 1, 128))
model.add(LSTM(64))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_data, epochs=10)
model.save_weights('rnn')
model.load_weights('rnn')
model.evaluate(test_data)

# Should print a very high score like 0.98.
print(model.predict([
  "i loved it! highly recommend it to anyone and everyone looking for a great movie to watch.",
]))

# Should print a very low score like 0.01.
print(model.predict([
  "this was awful! i hated it so much, nobody should watch this. the acting was terrible, the music was terrible, overall it was just bad.",
]))

