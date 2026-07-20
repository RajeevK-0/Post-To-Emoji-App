import streamlit as st
import pickle 
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re


st.set_page_config(page_title="Text To Emoji App", page_icon="🤖")
st.title("🖹➡️😄 Text-to-Emoji-App")
st.write("Type a sentence below, and the BiLSTM model will predict the sentiment!")

@st.cache_resource
def load_ml_artifacts():
    try:
        with open('MlArtifacts/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = load_model('MlArtifacts/emojiBiLSTM.keras') 
        return tokenizer, model
    except Exception as e:
        print("Error while loading ml artifacts:{e}")

tokenizer,model = load_ml_artifacts()
emoji_dictionary = {
'0'	:'❤️',	 '1':'😍',	 	
'2':'😂',	 '3':'💕',	
'4':'🔥',    '5':'😊',
'6':'😎',	 '7':'✨',
'8':'💙',    '9':'😘',
'10':'📷',   '11':'🌏',
'12':'☀️',   '13':'💜',
'14':'😉',   '15':'💯',	
'16':'😁',	 '17':'🎄',
'18':'📸',   '19':'😜'
}

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message['role'] == 'user':    
            st.markdown(message["content"])
        else:
            st.write(message["content_text"])
            cols = st.columns(5)
            for i, item in enumerate(message["content"]):
                with cols[i]:
                    st.metric(label=f"Rank {i+1}", value=item["emoji"], delta=f"{item['confidence']:.1f}%")

if user_input := st.chat_input("How are you feeling today?"):
    st.chat_message("user").markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    user_input = re.sub(r'[^a-zA-Z]','',user_input)

    sequence = tokenizer.texts_to_sequences([str(user_input)])
    padded_sequence = pad_sequences(sequence, maxlen=40)

    prediction_probs = model.predict(padded_sequence)[0]
    top5index = np.argsort(prediction_probs)[-5:][::-1]

    with st.chat_message("assistant"):
        intro_txt = "Here are the top 5 predicted emojis for your sentence:"
        st.write(intro_txt)
        cols = st.columns(5)
        response = []
        for i, class_idx in enumerate(top5index):
            emoji = emoji_dictionary.get(str(class_idx), "❓") 
            confidence = prediction_probs[class_idx] * 100 # 
            with cols[i]:
                st.metric(label=f"Rank {i+1}", value=emoji, delta=f"{confidence:.1f}%")
            response.append({'emoji':emoji,'confidence':confidence})
    
    st.session_state.messages.append({"role": "assistant","content_text":intro_txt ,"content": response})
if __name__ == "__main__":
   pass