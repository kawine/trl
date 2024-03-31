from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer
from datasets import Dataset

ref_model = AutoModelForCausalLM.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

dummy_dataset_dict = {
    "prompt": [
        "Hey, hello",
        "How are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "completion": [
        " hi nice to meet you",
        " leave me alone",
        " I don't have a name",
        " My name is Mary",
        " Python",
        " C++",
        " Java",
    ],
    "label": [
        True,
        False,
        False,
        True,
        True,
        False,
        False,
    ],
}
# fmt: on
dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

training_args = KTOConfig(
    output_dir='./test',
    per_device_train_batch_size=2,
    max_steps=3,
    remove_unused_columns=False,
    gradient_accumulation_steps=1,
    learning_rate=9e-1,
    evaluation_strategy="steps",
    beta=0.1,
)

trainer = KTOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dummy_dataset,
)

trainer.tokenize_row({"prompt": 'How are you', "completion": 'leave me alone', "label": True})