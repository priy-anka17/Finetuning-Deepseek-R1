# !pip  install unsloth
# !pip install --force-reinstall ---no-cache-dir ---no-deps git+https://github.com/unslothai/unsloth.git

from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))

from unsloth import FastLanguageModel
max_seq_length=2047
dtype=None
load_in_4bit=True

model, tokenizser=FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=userdata.get('HF_TOKEN')
)

model=FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

train_prompt_style="""Below is an instruction that describes a task,  paired with an input that provides further context. Write a response that appropriately completes the request.Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment plannig.
Please answer the following medical question.

### Question:
{}


### Response:
<think>
{}
</think>
{}"""

EOS_TOKEN=tokenizser.eos_token

def formatting_prompts_func(examples):
  inputs=examples["Question"]
  cots=examples["Complex_CoT"]
  outputs=examples["Response"]
  texts=[]
  for input, cot, output in zip(inputs, cots, outputs):
    text=train_prompt_style.format(input, cot, output)+EOS_TOKEN
    texts.append(text)
  return {
      "text":texts,
  }


from datasets import load_dataset
dataset=load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:100]", trust_remote_code=True)
dataset=dataset.map(formatting_prompts_func, batched=True,)

dataset["text"][0]

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer=SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizser,
    dataset_num_proc=2,

    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="outputs",
        report_to="none"
    ),
)

trainer_stats=trainer.train()

train_style="""Below is an instruction that describes a task,  paired with an input that provides further context. Write a response that appropriately completes the request.Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment plannig.
Please answer the following medical question.

### Question:
{}


### Response:
<think>
{}
</think>
{}"""

question="A 61-year-old woman with a loong history of involuntary urine loss during activities like caughing and sneezing but no leakage at night undergoes gynecological exam and Q-tip test. "
FastLanguageModel.for_inference(model)
inputs=tokenizer([prompt_style.format(question,"")], return_tensors="pt").to("cuda")

outputs =mode.generate(
    inpit_ids=input.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=true,
)
response =tokenizer.batch_decode(outputs)
print(response[0].split("###Response:") [1])

if False: model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")

if False: model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")

if False: model.save_pretrained_merged("model", tokenizer, save_method="lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="lora", token="")



