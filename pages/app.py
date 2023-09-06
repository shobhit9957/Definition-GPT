import os
from apikey import api_key
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = api_key

#app framework

st.title("Definition's GPT")
prompt = st.text_input("Plug in your prompt")

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='''You're a professional definiton's writer with a minimum of 5 years of experience.
    the user will give you a topic and you have to give the user 3 short line description about the topic.
    ex: 
    user: computer
    you:
    1) Computer: A programmable electronic device capable of performing various tasks by executing instructions, processing data, and generating output.
    2) Computer: A machine that stores, retrieves, and processes data using a combination of hardware components and software programs.
    3) Computer: An advanced tool that manipulates information through calculations, logic operations, and data storage, revolutionizing communication, work, and entertainment.

    When the user prompts you, first check that the prompt that is prompted by the user has an definiton or not, Remember you're only task is too, give definition's about a topic that is given by the user, You're not supposed to answer any other questions that are being requested by the user, you're only supposed to give definiions. if the prompt starts with "how to ", "what is" , "how about" first check, what the user is trying to say, if he is asking for any definition give him or if the user is just trying to play around and not asking definition's just say : 
    sorry, I'm definition's gpt, my only task is to provide definition's I can't provide any other stuff than that.
    Act like a friendly chat bot to the user, after completion of every task, greet them.

    
    Remember all the instructions I gave you and perform your task well!
    user : {topic}
    you:  
'''
)

#llm
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

if prompt: 
    response = title_chain.run(prompt)
    st.write(response)