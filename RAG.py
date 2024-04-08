from transformers import pipeline, Conversation
from transformers import AutoTokenizer, RagRetriever, TFRagModel
from fastapi import FastAPI
import os
from pydantic import BaseModel
import torch 




app = FastAPI()

class Article(BaseModel):
    text: str

class Conversation_return(BaseModel):
    conv_return: str

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
retriever = retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
generator = TFRagModel.from_pretrained("facebook/rag-token-base", retriever=retriever, from_pt=True)


@app.post("/", response_model=Conversation_return)
async def make_conversation(article: Article):
    input_text = article.text
    
    # Retrieve relevant passages using the RAG retriever
    retrieved_docs = retriever(input_text)
    
    # Generate a response using the RAG generator
    rag_inputs = tokenizer(input_text, return_tensors="pt")
    generated = generator.generate(
        rag_inputs["input_ids"],
        retrieved_docs=retrieved_docs,
        num_return_sequences=1,
        max_length=100,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    # Extract and return the generated response
    response_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    return {"conv_return": response_text}