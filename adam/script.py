from flashrag.config import Config
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate


def get_config(config_dict):
    return Config('my_config.yaml', config_dict=config_dict)

def get_prompt_template(config):
    system_prompt_rag = "You are a friendly AI Assistant." \
                    "Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable." \
                    "\nThe following are provided references. You can use them for answering question. Your answer should be clear and concise.\n\n{reference}"
    base_user_prompt = "{question}"
    return PromptTemplate(config, system_prompt=system_prompt_rag, user_prompt=base_user_prompt)

if __name__ == "__main__":
    # Fixed params
    use_case = "sa"
    temperature = 0
    topk = 3
    max_new_tokens  =2048
    
    query = ["What is the capital of France?"]
    
    config_dict = {"save_note":"demo",
                    'model2path': {'e5': 'intfloat/e5-base-v2', 'llama3-8B-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct'},
                    "retrieval_method":"e5",
                    'generator_model': 'llama3-8B-instruct',
                    "corpus_path":f"indexes/index/{use_case}/{use_case}.jsonl",
                    "index_path":f"indexes/index/{use_case}/e5_Flat.index"}
    config = get_config(config_dict)
    prompt_template = get_prompt_template(config)
    
    retriever = get_retriever(config)
    generator = get_generator(config)
    
    retrieved_docs = retriever.search(query,num=topk)
    retrieved_docs = retrieved_docs[0]
    
    input_prompt_with_rag = prompt_template.get_string(question=query, retrieval_result=retrieved_docs)
    response_with_rag = generator.generate(input_prompt_with_rag, 
                                            temperature=temperature, 
                                            max_new_tokens=max_new_tokens)[0]
    
    
    