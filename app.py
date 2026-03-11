import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please type a message before predicting.")
    else:
        # Vectorize the raw text directly
        vector_input = tfidf.transform([input_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Output
        if result == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")