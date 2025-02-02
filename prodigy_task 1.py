from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset_texts = ["This is an example sentence.", "GPT-2 can generate text."]

encoded_texts = tokenizer(dataset_texts, return_tensors="pt", padding=True, truncation=True)

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
attention_mask = inputs['attention_mask']

output = model.generate(input_ids, 
                        max_length=200,  
                        do_sample=True,    
                        temperature=0.7,    
                        top_k=50,          
                        top_p=0.9,         
                        pad_token_id=tokenizer.pad_token_id,  
                        attention_mask=attention_mask)  

print(tokenizer.decode(output[0], skip_special_tokens=True))
