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

class RAGChat:
    def __init__(self, index_name="faiss_index", model_id="google/gemma-2-9b-it"):
        self.index_name = index_name
        self.model_id = model_id
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.vectorstore = None
        self.qa_chain = None
        
        if not self.api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found.")
            
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.index_name):
            print(f"Warning: Vector store '{self.index_name}' not found.")
            return

        embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=self.api_key,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore = FAISS.load_local(
            self.index_name, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Setup LLM
        endpoint = HuggingFaceEndpoint(
            repo_id=self.model_id,
            huggingfacehub_api_token=self.api_key,
            temperature=0.3,
            max_new_tokens=512,
            task="text-generation"
        )
        llm = ChatHuggingFace(llm=endpoint)

        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS cite the source page number from the context at the end of your answer in the format [Page X].

Context:
{context}

Question: {question}
Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def ask(self, query: str):
        if not self.qa_chain:
            # Try to re-initialize if index was created after class instantiation
            self._initialize()
            if not self.qa_chain:
                return {"result": "Vector store not initialized. Please upload a PDF first.", "sources": []}

        try:
            result = self.qa_chain.invoke({"query": query})
            answer = result.get("result", "No answer generated.")
            source_docs = result.get("source_documents", [])
            
            sources = []
            for doc in source_docs:
                page = doc.metadata.get('page', 'Unknown')
                sources.append(f"Page {page + 1}")
                
            return {"answer": answer, "sources": list(set(sources))}
        except Exception as e:
            return {"error": str(e)}

def start_chat():
    print("--- RAG-Engine CLI ---")
    chat = RAGChat()
    
    while True:
        query = input("\nYou: ")
        if not query or query.lower() in ['exit', 'quit']:
            break
        
        print("AI is thinking...")
        response = chat.ask(query)
        
        if "error" in response:
            print(f"Error: {response['error']}")
        else:
            print(f"AI: {response['answer']}")
            print(f"Sources: {', '.join(response['sources'])}")

if __name__ == "__main__":
    start_chat()

if __name__ == "__main__":
    start_chat()
