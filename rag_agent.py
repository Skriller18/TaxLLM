import os
import torch
from huggingface_hub import login, hf_hub_download

# --- Critical Monkey Patch ---
import huggingface_hub
from huggingface_hub import hf_hub_download as original_hf_hub_download

if not hasattr(huggingface_hub, "cached_download"):
    def cached_download_wrapper(*args, **kwargs):
        kwargs.pop("url", None)
        kwargs.pop("legacy_cache_layout", None)
        return original_hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = cached_download_wrapper
# -----------------------------

from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from dotenv import load_dotenv
load_dotenv()

class TaxLawRAGSystem:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config["save_dir"], exist_ok=True)
        
        # Authenticate with Hugging Face
        login(token=self.config["hf_token"])

    def process_documents(self):
        """Load and split PDF documents"""
        loader = PyPDFLoader(self.config["pdf_path"])
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=["\n\n", "\nSection", "\nArticle", ". ", " ", ""],
            add_start_index=True
        )
        return text_splitter.split_documents(pages)

    def create_vector_db(self, docs):
        """Create FAISS vector store with modern embeddings"""
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"],
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vector_db = FAISS.from_documents(docs, embeddings)
        db_path = os.path.join(self.config["save_dir"], "taxlaw_faiss")
        vector_db.save_local(db_path)
        return vector_db

    def load_llm(self):
        """Load quantized Llama 3 with modern configurations"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config["llm_name"],
            use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config["llm_name"],
            quantization_config=bnb_config,
            device_map="auto",
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

        return HuggingFacePipeline(pipeline=text_generation_pipe)

    def create_rag_chain(self, vector_db):
        """Create enhanced RAG chain with system prompt"""
        system_prompt = (
            "You are a expert Indian Tax Law assistant. "
            "Always answer based on the provided legal context. "
            "Provide clear explanations with section references when available. "
            "If you're unsure about any information, clearly state this."
        )

        prompt_template = (
            "<|system|>\n{system_prompt}</s>\n"
            "<|user|>\n"
            "Context: {context}\n"
            "Question: {question}</s>\n"
            "<|assistant|>"
        )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["system_prompt", "context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.load_llm(),
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
            return_source_documents=True
        )

if __name__ == "__main__":
    config = {
        "pdf_path": "data/CA.pdf",
        "save_dir": "taxlaw_db",
        "chunk_size": 1536,
        "chunk_overlap": 256,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "llm_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "hf_token": os.getenv("HF_TOKEN")  # Load from .env file
    }

    tax_rag = TaxLawRAGSystem(config)

    print("Processing documents...")
    docs = tax_rag.process_documents()
    print(f"Created {len(docs)} document chunks")

    print("Creating vector database...")
    vector_db = tax_rag.create_vector_db(docs)

    print("Initializing RAG chain...")
    qa_chain = tax_rag.create_rag_chain(vector_db)

    query = "Explain the tax deductions available under Section 80C"
    response = qa_chain.invoke({"query": query, "system_prompt": "You are a expert Indian Tax Law assistant."})

    print("\nAnswer:", response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        print(f"- Page {page}: {snippet}...")

    print(f"\nVector database saved in: {config['save_dir']}")