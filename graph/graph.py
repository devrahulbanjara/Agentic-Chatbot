from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# Loading the env variables
load_dotenv()

# State variable
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# Defining our LLM (llama-3.1-8b-instant) from Groq for faster inference
llm = ChatGroq(model="llama-3.1-8b-instant")

# Chat node in our graph, that communicates to the LLM
def chat_node (state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(ChatState)

# in RAM memory for out chatbot
checkpointer = MemorySaver()

# Nodes in our graph
graph.add_node("chat_node", chat_node)

# Edges connecting the nodes
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


if __name__=="__main__":
    # Client Side
    thread_id = "1"
    
    while True:
        user_message = input("Type Here: ")
        print("User: ",user_message)
        
        if user_message.strip().lower() in ["end", "exit", "bye", "sakyo"]:
            break
        
        # Config variable that manages the persistence
        
        config = {"configurable" : {"thread_id" : thread_id}}
        
        response = chatbot.invoke({"messages" : [HumanMessage(content=user_message)]}, config=config)
        
        print("AI: ",response["messages"][-1].content)
        print()