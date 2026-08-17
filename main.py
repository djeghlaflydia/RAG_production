'''
LangChain is a framework for building applications that use LLMs like GPT, Claude, Llama, Gemini, etc.

The easiest way to understand it is:
🧠 LLM = the brain
The AI model that understands your input and generates responses.
🏢 OpenAI = the company/provider
The company that develops and provides AI models such as GPT through its API.
🤖 GPT = OpenAI's AI model family
The actual LLMs developed by OpenAI, such as GPT-4o and GPT-5.
🏢 Anthropic = the company/provider
The company that develops and provides AI models such as Claude through its API.
🤖 Claude = Anthropic's AI model family
The actual LLMs developed by Anthropic, such as Claude 3, Claude 3.5, and Claude 4.
🔗 LangChain = the framework
It helps your application communicate with and orchestrate different LLMs, whether they're from OpenAI, Anthropic, Google, or other providers.

                 YOUR RAG                             
                    │
                    ▼
              ┌───────────┐
              │   LLM     │
              └───────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     GPT          Claude       Llama
   OpenAI       Anthropic       Meta

ex. If you use GPT: Your computer → Internet → OpenAI servers → GPT


in my project:
                     YOUR RAG
                         │
                      LangChain
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           Gemini      Groq       Ollama
           Google      Llama       Llama



Python
  │
  ├── LangChain ──────── LLM + RAG components
  │
  ├── LangGraph ──────── RAG workflow / agents
  │
  ├── OpenAI / Claude ── LLM
  │
  └── Vector DB ──────── Document retrieval

'''

from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {lg_version}")


def main():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Gemini LLM Response: {response.content}")

    # test the Groq LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Groq LLM Response: {response.content}")

    print("Setup complete!")

if __name__ == "__main__":
    main()


