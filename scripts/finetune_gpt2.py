from datasets import load_dataset
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import login
import torch
from transformers import DataCollatorForLanguageModeling

hf_token = ""
login(token=hf_token)

custom_model_name = "<Enter custom model name>"
dataset_name = "<Enter dataset name>"
split_name = "general"  # real_baseline / real_mild / real_moderate / real_severe
text_column = "text"   # column to be trained on (text, florence_caption, qwen_audiio_caption)

def fine_tune_on_hf(real_dataset_name, split):
    # Load dataset from Hugging Face Hub
    train_dataset = load_dataset(real_dataset_name, split=split, use_auth_token=True)
    columns_to_keep = [text_column]
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_keep])
    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test
    # Get the train and test sets
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        device_map="auto",
    )

    # Set padding token (GPT-2 does not have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Labels should be identical to input_ids
        return tokenized

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Use a DataCollator for proper batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, so no masked language modeling
    )

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using ", device)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=custom_model_name,
        eval_strategy="epoch",
        learning_rate=2e-5,
        warmup_steps=1000,
        per_device_train_batch_size=8,  # Reduce if memory issues persist
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.001,
        push_to_hub=True,
        report_to="tensorboard",
        logging_dir=f"{custom_model_name}/logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=True  # Enable mixed precision

    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,  # Replace deprecated tokenizer argument
    )

    # Train the model
    trainer.train()

    # Save and push the trained model
    trainer.save_model(custom_model_name)
    trainer.push_to_hub(custom_model_name)


fine_tune_on_hf(dataset_name, split_name)  # Fine-tune GPT-2


