import os
import chainlit as cl
#from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

model_id = "gpt2-medium"
conv_model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id, 
                            model_kwargs={"temperature":0.6, "max_new_tokens":150})

template = """You are an AI story writer assistant that completes a story based on the input query received. {query}"""

@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Perform post-processing on the received response here
    # res is a dictionary and the response text is stored under the key "text"
    await cl.Message(content=res["text"]).send()