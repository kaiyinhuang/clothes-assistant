from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 加载微调后的模型
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 创建生成管道
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# 生成回复
def generate_response(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    response = chatbot(
        prompt,
        max_length=512,
        temperature=0.7,
        num_return_sequences=1,
    )
    assistant_text = response[0]["generated_text"].split("Assistant:")[-1].strip()
    return assistant_text

# 示例
if __name__ == "__main__":
    user_query = "I need a professional dress for an interview."
    print("User:", user_query)
    print("Assistant:", generate_response(user_query))
