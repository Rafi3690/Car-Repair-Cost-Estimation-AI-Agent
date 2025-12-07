import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
from langchain_deepseek import ChatDeepSeek
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from ddgs import DDGS

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

system_prompt = """
You are a professional Car Repair Cost Range Assistant.

Your task:
- Only provide approximate cost ranges for car repairs.
- User will always mention:
    1. Size of the damage in inches.
    2. Location (city or country) where repair will happen.
- Your answer must be in **plain English, ONE concise line**.
- Do NOT extract or search for numerical patterns using regex.
- Do NOT answer unrelated queries; if the query is not about car repair, politely decline.
- If exact cost is not available, provide an **estimated range** based on typical repair costs for similar cars and damage size.
- Include the car model if mentioned by the user.

Example output:
- Input: "BMW X5 M 30 inch dent repair in Canada"
- Output: "Repairing a 30-inch dent on a BMW X5 M in Canada typically costs between $800 and $1500."

Constraints:
- Always keep it **short, clear, and professional**.
- Avoid explanations or extra commentary.
"""

class WebAssistant:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        Web_Search_Tool = Tool(
            name="Web_Search_tool",
            func=lambda query, top_k=500: self.search(query, top_k),
            description=f"Search real-time info using DDGS and return results in plain natural language. {system_prompt}"
        )

        self.tools = [Web_Search_Tool]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )   
    
    def search(self, query, top_k=500):
        """
        Perform a real-time search using DDGS and return raw results.

        Parameters:
        - query (str): The search query string.
        - top_k (int): Maximum number of search results to retrieve.

        Returns:
        - dict: {
            "query_used": original search query,
            "results": list of raw search result snippets
        }

        Note:
        - This method does NOT use regex.
        - Results are raw text snippets suitable for LLM to process into natural language answers.
        """
        results = DDGS().text(query, max_results=top_k)
        return {
            "query_used": query,
            "results": results
        }


    def humanize(self, info):
        if isinstance(info, dict) and "output" in info:
            return info["output"]
        return str(info)
    
    def run(self, query):
        prompt = f"{query}\nPlease answer in ONE concise line."
        raw_output = self.agent.invoke({"input": prompt})
        return self.humanize(raw_output)


def create_agent():
    llm = ChatDeepSeek(
        api_key=deepseek_api_key,
        model="deepseek-chat",
        # model="deepseek-resoaner",
        temperature=0.0,
        max_tokens=256,
        top_p=1.0,
        verbose=True
    )
    return WebAssistant(llm)


if __name__ == "__main__":
    bot = create_agent()
    try:
        while True:
            user_input = input("You: ")
            output = bot.run(user_input)
            print("Bot:", output)
    except KeyboardInterrupt:
        print("\nStopped.")
