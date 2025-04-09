import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the LSTM Model
model = load_model('next_word_lstm.keras')

## Load the tokenizer
with open('tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

# Function to Predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_len:
    token_list = token_list[-(max_sequence_len):]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
  predicted = model.predict(token_list, verbose = 0)
  predicted_word_index = np.argmax(predicted,axis = 1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

## Streamlit app
st.title("Next Word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter a sequence of word:","To be or not to be")
if st.button("Predict"):
  max_sequence_len=model.input_shape[1]
  next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
  st.write(f"The next word is: {next_word}")
