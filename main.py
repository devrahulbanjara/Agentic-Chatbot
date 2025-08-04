from fastapi import FastAPI
from graph.graph import chatbot
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

app=FastAPI()

@app.get("/")
def home():
    return {"detail": "Welcome to the api of Agentic Chatbot using LangGraph"}

class Request(BaseModel):
    user_input: str
    thread_id: str

@app.("/chat")
async def chat(request: Request):
    CONFIG = {'configurable': {'thread_id': request.thread_id}}
    
    response = chatbot.invoke({'messages': [HumanMessage(content=request.user_input)]}, config=CONFIG)
    
    return {"response": response['messages'][-1].content}