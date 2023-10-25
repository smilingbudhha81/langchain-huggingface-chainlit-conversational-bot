import os
import chainlit as cl
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import LLMChain
from langchain.handlers import AsyncLangchainCallbackHandler

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face API client
api = HfApi()

# Set the ID of the model you want to use, such as "gpt2-medium"
model_id = "<MODEL_ID_HERE>"

# Use the API client to download the tokenizer and model files
model_info = api.model_info(model_id)
model_details = api.model_details(model_info.modelId)
model_files = api.model_files(model_info.modelId, revision=model_details.currentRevision)
tokenizer = AutoTokenizer.from_pretrained(model_files[0].download_url)
model = AutoModelForCausalLM.from_pretrained(model_files[1].download_url)
llm_chain = LLMChain(tokenizer=tokenizer, model=model, max_length=model.config.max_position_embeddings)

# Set up the prompt for the chatbot
template = "You are an AI story writer assistant that completes a story based on the input query received. {query}"

@cl.on_message
async def main(message: str):
    # Get the LangChain session and the LLM chain
    langchain_session = cl.user_session.get("langchain_session")
    llm_chain = langchain_session.get("llm_chain")
    
    # Use the message text as the input to the LLM chain
    response = await llm_chain.acall(message, callbacks=[AsyncLangchainCallbackHandler()])
    
    # Get the generated text from the response
    generated_text = response["text"]
    
    # Send the generated text as a message reply
    await cl.Message(content=generated_text).send()

