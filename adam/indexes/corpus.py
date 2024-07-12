from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from flashrag.retriever.index_builder import Index_Builder
import os
import json

def load_pdf(data_dir):
    dataset = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            dataset.extend(pages)
    return dataset

def init_embeddings(model_path):
    modelPath = model_path
    model_kwargs = {'device':'cuda'}
    encode_kwargs = {'normalize_embeddings': True}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    return HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

def create_corpus(docs):
    # Example: {"id": "0", "contents": "contents for building index"}
    jsonl = []
    for i, doc in enumerate(docs):
        dictionary = {}
        content = doc.page_content
        dictionary["id"] = str(i)
        dictionary["contents"] = content
        jsonl.append(dictionary)
    return jsonl

def write_to_jsonl(file_path, corpus):
    # Write data to .jsonl file
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in corpus:
            jsonl_file.write(json.dumps(entry) + '\n')

def build_index(jsonl_file_path, index_file_path):
    index_builder = Index_Builder(
        retrieval_method="e5",
        model_path="intfloat/e5-base-v2",
        corpus_path=jsonl_file_path,
        save_dir=index_file_path,
        use_fp16=True,
        max_length=200,
        batch_size=32,
        pooling_method="mean",
        faiss_type="Flat"
    )
    index_builder.build_index()

def main():
    use_cases = os.listdir("/home/adamtay/FlashRAG/adam/indexes/documents")
    model_path = "intfloat/e5-base-v2"
    embeddings = init_embeddings(model_path)
    for use_case in use_cases:
        print(f"Loading {use_case}")
        pdf_dir = f"/home/adamtay/FlashRAG/adam/indexes/documents/{use_case}"
        dataset = load_pdf(pdf_dir)
        contents = [data.page_content for data in dataset ]
        
        # Split documents
        print(f"Splitting {use_case}")
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        docs = text_splitter.create_documents(contents)
        
        # Format to jsonl
        print(f"Creating corpus for {use_case}")
        corpus = create_corpus(docs)
        print(corpus)
        
        # Write to jsonl
        print(f"Building index for {use_case}")
        jsonl_file_path = f"index/{use_case}/{use_case}.jsonl"
        write_to_jsonl(jsonl_file_path, corpus)
        index_file_path = f"index/{use_case}"
        build_index(jsonl_file_path, index_file_path)
        print(f"Index saved to {index_file_path}")
    
main()