# app.py

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory  import ConversationBufferMemory


# Load Embeddings instance
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


# Load Existing Chroma DB created
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = db.as_retriever(search_kwargs={"k": 4})


# LLM 
llm = ChatOllama(model="llama3.1")


#  Memory 
memory = ConversationBufferMemory(return_messages=True)


#  Prompt  for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Answer clearly and concisely.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])



while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        break


    docs = retriever.invoke(query)

    
    context = "\n\n".join([doc.page_content for doc in docs])

    #  Load Chat History 
    chat_history = memory.load_memory_variables({})["history"]

    # Create Prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history
    })

    # Get Response 
    response = llm.invoke(final_prompt)

    print("\nBot:", response.content, "\n")

    # Save to Memory 
    memory.save_context(
        {"input": query},
        {"output": response.content}
    )