import os
import time
import logging
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("taxllm.log")
    ]
)
logger = logging.getLogger("TaxLLM")

# Load environment variables
load_dotenv()

app = FastAPI()

logger.info("Starting TaxLLM server")

# Configuration for model and data
config = {
    "pdf_path": "data/CA.pdf",
    "save_dir": "taxlaw_db",
    "chunk_size": 1536,
    "chunk_overlap": 256,
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "llm_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "hf_token": os.getenv("HF_TOKEN")
}

# Authenticate with Hugging Face
login(token=config["hf_token"])
logger.info(f"Authenticated with Hugging Face")

# Define the path for the saved FAISS index
vector_db_path = os.path.join(config["save_dir"], "taxlaw_faiss")

# Load or create the vector store
if os.path.exists(vector_db_path):
    logger.info("Loading existing vector database...")
    vector_db = FAISS.load_local(
        vector_db_path,
        HuggingFaceEmbeddings(
            model_name=config["embedding_model"],
            model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        ),
        allow_dangerous_deserialization=True
    )
else:
    logger.info("Processing documents and creating vector database...")
    loader = PyPDFLoader(config["pdf_path"])
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\nSection", "\nArticle", ". ", " ", ""],
        add_start_index=True
    )
    docs = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"],
        model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(vector_db_path)
    logger.info(f"Vector database created and saved to {vector_db_path}")

# Load the LLM with quantization settings
logger.info(f"Loading LLM model: {config['llm_name']}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(config["llm_name"], use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config["llm_name"],
    quantization_config=bnb_config,
    device_map={"": "cuda:0"},
    attn_implementation="flash_attention_2"
)

text_generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15,
    return_full_text=False
)
logger.info("Text generation pipeline created")

# Subclass HuggingFacePipeline so that input_variables is declared as a field (recognized by Pydantic)
class CustomHuggingFacePipeline(HuggingFacePipeline):
    input_variables: list = Field(default_factory=lambda: ["question", "context"])

llm_pipeline = CustomHuggingFacePipeline(pipeline=text_generation_pipe)

# Build the prompt with a system message.
system_prompt = (
    "You are an expert Indian Tax Law assistant. "
    "Always answer based on the provided legal context. "
    "Provide clear explanations with section references when available. "
    "If you're unsure about any information, clearly state this."
)
prompt_template = (
    f"<|system|>\n{system_prompt}</s>\n"
    f"<|user|>\n"
    f"Context: {{context}}\n"
    f"Question: {{question}}</s>\n"
    f"<|assistant|>"
)

# Explicitly set the input_variables so the prompt knows to expect 'context' and 'question'
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
logger.info("Prompt template created with input variables: %s", prompt.input_variables)

# Create the RetrievalQA chain.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_pipeline,
    chain_type="stuff",
    retriever=vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    ),
    chain_type_kwargs={
        "prompt": prompt,
        "document_separator": "\n\n",
        "document_variable_name": "context"
    },
    return_source_documents=True,
    input_key="question",
    output_key="result"
)
logger.info("RetrievalQA chain created successfully")

# Define request model for FastAPI using 'question' as the key.
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    question = request.question
    start_time = time.time()
    
    logger.info(f"Received question: '{question}'")
    
    try:
        logger.info("Processing question...")
        response = qa_chain.invoke({"question": question})
        
        # Extract a brief snippet from each source document for reference
        sources = []
        for doc in response.get("source_documents", []):
            page = doc.metadata.get("page", "N/A")
            snippet = doc.page_content[:150].replace("\n", " ").strip()
            sources.append({"page": page, "snippet": snippet})
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Response generated in {processing_time:.2f} seconds")
        return {"result": response.get("result", ""), "sources": sources, "processing_time": f"{processing_time:.2f}s"}
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return {"error": str(e), "processing_time": f"{processing_time:.2f}s"}

if __name__ == "__main__":
    logger.info("Starting server at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)