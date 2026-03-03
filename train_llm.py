import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

model_name = "google/flan-t5-small"  # smaller for CPU

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("empathetic_dialogues", split="train[:2000]")

def preprocess(example):
    input_text = "Provide emotional support: " + example["utterance"]
    target_text = example["utterance"]

    model_inputs = tokenizer(
        input_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        target_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./maitri_llm",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./maitri_llm")
tokenizer.save_pretrained("./maitri_llm")