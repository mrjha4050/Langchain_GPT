# bringing dependencies 
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from dotenv import load_dotenv
load_dotenv()


# App framework
st.title('💎 GPT For Newbies')
prompt = st.text_input('Plug in your question here')

# Prompt Template 
title_template = PromptTemplate(
    input_variables=['topic'],
    template='give article or youtube video refernce {topic}'
)
script_template = PromptTemplate(
   input_variables = ['title', 'wikipedia_research'], 
    template=' explain in simple language {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(openai_api_key =os.getenv[api_key], temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)    

wiki = WikipediaAPIWrapper()

# Displaying information
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
