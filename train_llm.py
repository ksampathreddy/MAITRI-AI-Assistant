import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)

# ===============================
# 1️⃣ Device Setup
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ===============================
# 2️⃣ Load SMALL Model (FAST)
# ===============================
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

# ===============================
# 3️⃣ Load Dataset
# ===============================
dataset = load_dataset(
    "empathetic_dialogues",
    split="train",
    trust_remote_code=True
)

# 🔥 Reduce dataset for speed
dataset = dataset.select(range(2000))

print("Dataset loaded:", len(dataset))

# ===============================
# 4️⃣ Build Conversation Pairs
# ===============================
def build_pairs(dataset):

    inputs = []
    targets = []

    for i in range(len(dataset) - 1):

        current = dataset[i]
        next_utt = dataset[i + 1]

        if current["conv_id"] == next_utt["conv_id"]:
            if current["speaker_idx"] != next_utt["speaker_idx"]:

                input_text = f"""
You are MAITRI, an emotional support AI for astronauts.

Emotion context: {current['context']}
User: {current['utterance']}

Provide a supportive response.
"""

                target_text = next_utt["utterance"]

                inputs.append(input_text)
                targets.append(target_text)

    return inputs, targets


inputs, targets = build_pairs(dataset)

print("Training pairs created:", len(inputs))

# ===============================
# 5️⃣ Create Dataset Object
# ===============================
training_data = Dataset.from_dict({
    "input_text": inputs,
    "target_text": targets
})

# ===============================
# 6️⃣ Tokenization
# ===============================
def tokenize_function(example):

    model_inputs = tokenizer(
        example["input_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["target_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = training_data.map(
    tokenize_function,
    remove_columns=training_data.column_names
)

# ===============================
# 7️⃣ Training Arguments (FAST)
# ===============================
training_args = TrainingArguments(
    output_dir="./maitri_llm",
    per_device_train_batch_size=8,   # bigger batch = faster on CPU
    num_train_epochs=1,              # 1 epoch for hackathon
    logging_steps=100,
    learning_rate=5e-5,
    fp16=False,
    save_total_limit=1,
    report_to="none"
)

# ===============================
# 8️⃣ Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# ===============================
# 9️⃣ Train
# ===============================
print("Starting FAST training...")
trainer.train()

# ===============================
# 🔟 Save Model
# ===============================
model.save_pretrained("./maitri_llm")
tokenizer.save_pretrained("./maitri_llm")

print("✅ FAST Fine-tuned MAITRI LLM saved successfully!")