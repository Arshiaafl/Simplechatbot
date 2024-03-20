from transformers import pipeline, Conversation
from fastapi import FastAPI
import os
from pydantic import BaseModel

app = FastAPI()

class Article(BaseModel):
    text: str

class Conversation_return(BaseModel):
    conv_return: str

Conversation_builder = pipeline("conversational", model="facebook/blenderbot-400M-distill")

@app.post("/", response_model=Conversation_return)
async def make_conversation(article:Article):
    my_text = Conversation(article.text)
    conv = Conversation_builder(my_text)
    return {"conv_return": conv.messages[-1]["content"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)