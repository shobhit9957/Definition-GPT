import os
import streamlit as st


#api key 
st.title("API key")
api_key = st.text_input("Please provide a valid openai api key here")
if api_key:
    st.write(f"Thanks for providing your API key.")