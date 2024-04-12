from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

import pandas as pd
import numpy as np
import torch

LOCATION = '.'

# GPT-3 모델과 토크나이저 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1', bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
max_length = 256

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token = '<s>'
        self.eos_token = '</s>'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data.iloc[idx]['text'])
        corrected_text = str(self.data.iloc[idx]['corrected'])

        # Tokenize input and output texts
        input_ids = self.tokenizer.encode(self.bos_token + input_text + self.eos_token, 
                                          max_length=self.max_length, truncation=True, padding='max_length')
        output_ids = self.tokenizer.encode(self.bos_token + corrected_text + self.eos_token, 
                                           max_length=self.max_length, truncation=True, padding='max_length')

        # Create attention mask for inputs (0s where padded, 1s elsewhere)
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(output_ids, dtype=torch.long)
        }


train = pd.read_csv(f'{LOCATION}/train.csv')
train.drop(columns=['id'], inplace=True)
validate = pd.read_csv(f'{LOCATION}/validate.csv')
validate.drop(columns=['id'], inplace=True)

train_dataset = TextDataset(train, tokenizer, max_length)
validate_dataset = TextDataset(validate, tokenizer, max_length)

from transformers import BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_metric

import torch
import pickle
import os
import gc

device = torch.device('cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
1
model.train()

lr = 2e-5
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
SAVE_PATH = f"{LOCATION}/all_metrics.pkl"
bleu_metric = load_metric("bleu")

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "rb") as f:
        all_metrics = pickle.load(f)
else:
    all_metrics = []

def compute_metrics(pred):
    predictions = pred.predictions.argmax(-1)
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in pred.label_ids]

    # BLEU score 계산
    bleu_score = bleu_metric.compute(predictions=[decoded_preds], references=[[label] for label in decoded_labels])

    scores = {
        "epoch": pred.metrics["epoch"],
        "train_loss": pred.metrics["train_loss"],
        "val_loss": pred.metrics["eval_loss"],
        "epoch": pred.metrics["epoch"],
        "bleu": bleu_score["bleu"],
    }
    all_metrics.append(scores)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(all_metrics, f)
    return scores

def get_latest_checkpoint(path):
    checkpoints = [f for f in os.listdir(path) if f.startswith('checkpoint-')]
    if checkpoints:
        return os.path.join(path, max(checkpoints))
    else:
        return None

data_collactor = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=f"{LOCATION}/model", # 모델 저장 경로
    overwrite_output_dir=True,  # 기존 모델을 덮어쓰기
    learning_rate=2e-5,  # 학습률 설정
    num_train_epochs=50,  # 학습 에포크 설정
    evaluation_strategy="epoch",  # 평가 스트레티지 설정
    per_device_train_batch_size=16,  # 배치 크기 설정
    per_device_eval_batch_size=16,
    save_steps=1000,  # 모델 저장 스텝 설정
    save_total_limit=10,  # 최대 모델 저장 개수 설정
    warmup_steps=500,  # 워밍업 스텝 설정
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    data_collator=data_collactor,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

torch.cuda.empty_cache()
gc.collect()

trainer.train(resume_from_checkpoint = True if get_latest_checkpoint(f'{LOCATION}/bart/') else False)


trainer.save_model(f"{LOCATION}/bart_model")
