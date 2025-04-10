from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import torch

# 1. 加载Hugging Face数据集
dataset = load_dataset("dvllasuero/clothes-assistant", split="train")
dataset = dataset.train_test_split(test_size=0.1)

# 2. 加载模型和分词器
model_name = "gpt2"  # 或 "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充符
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. 数据预处理（假设数据集已有 "messages" 字段）
def format_conversation(examples):
    text = ""
    for msg in examples["messages"]:
        text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return {"text": text}

def preprocess_function(examples):
    inputs = tokenizer(
        [format_conversation(ex)["text"] for ex in examples["messages"]],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. 定义训练参数
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_dir="./logs",
)

# 5. 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()

# 6. 保存模型
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
