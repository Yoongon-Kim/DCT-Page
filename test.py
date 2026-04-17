from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
out = tok.apply_chat_template(                                                                                                   
    [{'role':'user','content':'hello'}],
    add_generation_prompt=True, tokenize=False,                                                                                  
)               
print(repr(out))  