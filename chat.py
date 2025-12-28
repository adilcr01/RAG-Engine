import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
# from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv(r"..\.env")

def start_chat():
    print("Initializing Chat System...")
    
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        print("Error: HUGGINGFACEHUB_API_TOKEN not found.")
        return

    # Load Vector Store
    index_name = "faiss_index"
    if not os.path.exists(index_name):
        print(f"Vector store '{index_name}' not found. Please run ingest.py first.")
        return
        
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=api_key,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"Loading FAISS index from '{index_name}'...")
    try:
        vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return
    
    # Setup LLM (Using a model that explicitly supports the text-generation task on HF)
    # llm_model = "google/flan-t5-large"
    
    # print(f"Loading model: {llm_model}...")
    # llm = HuggingFaceEndpoint(
    #     repo_id=llm_model,
    #     huggingfacehub_api_token=api_key,
    #     temperature=0.3,
    #     max_new_tokens=512,
    #     # task="text-generation"  <-- Optional: Remove this line to let the library auto-detect
    # )
    
    llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print(f"Loading model: {llm_model}...")
    
    # 1. Connect to the Endpoint
    endpoint = HuggingFaceEndpoint(
        repo_id=llm_model,
        huggingfacehub_api_token=api_key,
        temperature=0.3,
        max_new_tokens=512,
        task="text-generation" # actually just remove this line or let it default, 
                               # but if it fails, try task="conversational" 
    )
    
    # 2. Wrap it in a Chat Interface so LangChain can handle it
    llm = ChatHuggingFace(llm=endpoint)


    # Custom Prompt to ensure source citation
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS cite the source page number from the context at the end of your answer in the format [Page X].
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    print("Initializing RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("\n--- AI Chat Lease Assistant ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nYou: ")
            if not query or query.lower() in ['exit', 'quit']:
                print("Exiting chat...")
                break
            
            print("AI is thinking...")
            
            # Step 1: Retrieval
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"Error during retrieval: {e}")
                continue
            
            # Step 2: LLM Generation
            try:
                result = qa_chain.invoke({"query": query})
                answer = result.get("result")
                source_docs = result.get("source_documents", [])
                
                if not answer:
                    print("AI: Sorry, I couldn't generate a response.")
                else:
                    print(f"AI: {answer}")
                    
                print("\nSources:")
                if not source_docs:
                    print("- No sources found.")
                for doc in source_docs:
                    page = doc.metadata.get('page', 'Unknown')
                    # pypdf page is 0-indexed, so add 1 for user friendly page
                    print(f"- Page {page + 1}")
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except BaseException as e:
            print(f"CRITICAL ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    start_chat()
