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
    # Try these models in order - first available will be used
    # Mistral is generally better at following instructions for RAG
    DEFAULT_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.2",  # Better instruction following
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Alternative Mistral
        "google/gemma-2-9b-it",  # Fallback to original
    ]
    
    def __init__(self, index_name="faiss_index", model_id=None):
        self.index_name = index_name
        # Use provided model or default
        self.model_id = model_id or self.DEFAULT_MODELS[0]
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

        # Use a more direct prompt format that works better with Gemma models
        prompt_template = """<start_of_turn>user
You are analyzing a document. Below is the EXACT content from the document that contains the answer to the question.

DOCUMENT CONTENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Read the document content above carefully
2. Find the answer to the question in the document content
3. Provide the answer using information ONLY from the document content
4. Include specific numbers, dates, amounts, or details from the document
5. If you find the answer, state it clearly
6. If the answer is not in the provided content, say "I cannot find this information in the provided document content"
7. NEVER say you don't have access to files - you have the document content above

ANSWER:<end_of_turn>
<start_of_turn>model"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 15}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def ask(self, query: str):
        # Check if vectorstore is loaded, re-initialize if needed
        if not self.vectorstore:
            self._initialize()
            if not self.vectorstore:
                return {"answer": "Document index not loaded. Please upload a PDF first.", "sources": []}

        try:
            
            # First, retrieve relevant documents
            # Increase k for better coverage when searching for multiple items
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
            source_docs = retriever.invoke(query)
            
            # Debug logging
            print(f"\n=== Query: {query} ===")
            print(f"Retrieved {len(source_docs)} source documents")
            
            if not source_docs:
                return {"answer": "No relevant content found in the document. Please try rephrasing your question.", "sources": []}
            
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(source_docs):
                page = doc.metadata.get('page', 'Unknown')
                content = doc.page_content
                context_parts.append(f"[Document excerpt from page {page + 1 if page != 'Unknown' else 'N/A'}]:\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Debug: Print first 500 chars of context
            print(f"Context preview (first 500 chars): {context[:500]}...")
            print(f"Using model: {self.model_id}")
            
            # Use the LLM directly with formatted context
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_id,
                huggingfacehub_api_token=self.api_key,
                temperature=0.1,  # Lower temperature for more focused answers
                max_new_tokens=1024,  # Increased for longer answers
                task="text-generation"
            )
            llm = ChatHuggingFace(llm=endpoint)
            
            # Determine prompt format based on model
            if "mistral" in self.model_id.lower():
                # Mistral format
                # Check if query is asking for multiple/all items
                is_extraction_query = any(word in query.lower() for word in ["all", "list", "every", "each", "multiple", "how many"])
                
                extraction_instructions = """
- If asked to find ALL items (like "all checks", "list all", "every check"), you MUST search through ALL the document content provided
- Extract EVERY instance mentioned in the document, not just one
- Provide a complete list or count of ALL items found
- If finding multiple items, format them clearly (numbered list or table format)
- Double-check the entire document content to ensure you haven't missed any instances""" if is_extraction_query else ""
                
                direct_prompt = f"""<s>[INST] You are a helpful assistant that answers questions based on provided document content.

DOCUMENT CONTENT:
{context}

QUESTION: {query}

IMPORTANT INSTRUCTIONS:
- You HAVE access to the document content shown above
- Answer the question using ONLY the information from the document content provided
- Extract and state specific numbers, amounts, dates, or details directly from the document
{extraction_instructions}
- If the answer is in the document, provide it clearly and accurately
- If the answer cannot be found in the provided content, say "I cannot find this information in the provided document content"
- NEVER say you don't have access to files or PDFs - the document content is provided above

Now answer the question based on the document content: [/INST]"""
            else:
                # Generic format (for Gemma and others)
                # Check if query is asking for multiple/all items
                is_extraction_query = any(word in query.lower() for word in ["all", "list", "every", "each", "multiple", "how many"])
                
                extraction_instructions = """
6. If asked to find ALL items (like "all checks", "list all", "every check"), you MUST search through ALL the document content provided
7. Extract EVERY instance mentioned in the document, not just one
8. Provide a complete list or count of ALL items found
9. If finding multiple items, format them clearly (numbered list or table format)
10. Double-check the entire document content to ensure you haven't missed any instances""" if is_extraction_query else ""
                
                direct_prompt = f"""You are a helpful assistant. Answer the question using ONLY the document content provided below.

DOCUMENT CONTENT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. You HAVE the document content above - use it to answer the question
2. Extract specific numbers, amounts, dates, or details from the document
{extraction_instructions}
3. If you find the answer, state it clearly
4. If not found, say "I cannot find this information in the provided document content"
5. NEVER say you don't have access to files - the document content is provided above

ANSWER:"""
            
            try:
                response = llm.invoke(direct_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # Clean up the answer (remove any model turn markers)
                answer = answer.strip()
                # Remove common chat template markers
                for marker in ["<start_of_turn>model", "<end_of_turn>", "</s>", "<s>", "[INST]", "[/INST]"]:
                    answer = answer.replace(marker, "").strip()
                
                # Check if model is refusing to answer (common issue)
                refusal_phrases = [
                    "i lack the capability",
                    "i don't have access",
                    "i cannot access",
                    "i am unable to",
                    "i do not have the ability"
                ]
                
                if any(phrase in answer.lower() for phrase in refusal_phrases):
                    print("WARNING: Model is refusing to answer. This might be a model limitation.")
                    print(f"Raw answer: {answer[:300]}")
                    # Try to extract any useful information anyway
                    if context and len(context) > 100:
                        answer = f"I found relevant content in the document. Here's what I can extract: {context[:500]}... Please review the document content above to find the specific answer."
                
            except Exception as llm_error:
                print(f"Error calling LLM: {llm_error}")
                # Fallback: return context directly
                answer = f"Error with model {self.model_id}. Here's the relevant document content I found:\n\n{context[:1000]}"
            
            # Extract sources
            sources = []
            for doc in source_docs:
                page = doc.metadata.get('page', 'Unknown')
                if page != 'Unknown':
                    sources.append(f"Page {page + 1}")
                else:
                    sources.append("Document")
            
            print(f"Answer generated: {answer[:200]}...")
            print(f"Sources: {sources}\n")
                
            return {"answer": answer, "sources": list(set(sources))}
        except Exception as e:
            print(f"Error in RAGChat.ask(): {e}")
            import traceback
            traceback.print_exc()
            return {"answer": f"Error processing query: {str(e)}", "sources": []}

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
