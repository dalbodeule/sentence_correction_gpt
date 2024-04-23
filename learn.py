from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from typing import Tuple

import pandas as pd
import numpy as np
import torch

LOCATION = '.'

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2', bos_token="<s>", eos_token="</s>")
max_length = 128

print(f"BOS_TOKEN: {tokenizer.bos_token}, EOS_TOKEN: {tokenizer.eos_token}")

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length, chunksize=3000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunksize = chunksize
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        
        self.reader = pd.read_csv(file_path)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        row = self.reader.iloc[idx]
        input_text = str(row['text'])
        corrected_text = str(row['corrected'])
        
        input_encodings = self.tokenizer.encode_plus(
            self.bos_token + input_text + self.eos_token,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        output_encodings = self.tokenizer.encode_plus(
            self.bos_token + corrected_text + self.eos_token,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = input_encodings['input_ids'].squeeze()
        attention_mask = input_encodings['attention_mask'].squeeze()
        output_ids = output_encodings['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids
        }

train_dataset = TextDataset(f'{LOCATION}/train.csv', tokenizer, max_length)
validate_dataset = TextDataset(f'{LOCATION}/validate.csv', tokenizer, max_length)

from transformers import BartForConditionalGeneration, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from evaluate import load

import torch
import pickle
import os
import gc

device = torch.device('cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
lr=2e-5
SAVE_PATH = f"{LOCATION}/all_metrics.csv"
metric_bleu = load("bleu")
metric_rouge = load("rouge")
metric_meteor = load("meteor")

if os.path.exists(SAVE_PATH):
    all_metrics = pd.read_csv(SAVE_PATH)
else:
    all_metrics = pd.DataFrame(columns=["bleu", "rouge", "meteor"])

def get_latest_checkpoint(path):
    checkpoints = [f for f in os.listdir(path) if f.startswith('checkpoint-')]
    if checkpoints:
        return os.path.join(path, max(checkpoints))
    else:
        return None

def compute_metrics(eval_pred: Tuple[torch.Tensor, torch.Tensor]):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.array(labels)
    neg_labels = labels < 0
    if np.any(neg_labels):
        labels[neg_labels] = 0

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = metric_bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    rouge_score = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_score = metric_meteor.compute(predictions=decoded_preds, references=decoded_labels)

    score = {"bleu": bleu_score["bleu"], "rouge": rouge_score['rougeL'], "meteor": meteor_score["meteor"]}

    all_metrics.loc[len(all_metrics)] = score
    all_metrics.to_csv(SAVE_PATH)

    return score

def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

class EmptyCacheCallback(TrainerCallback):
    """훈련의 각 로그 스텝마다 CUDA 캐시를 비우는 콜백"""

    def on_log(self, args, state, control, **kwargs):
        """로그 이벤트가 발생할 때 CUDA 캐시를 비웁니다."""
        gc.collect()
        torch.mps.empty_cache()

data_collactor = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    eval_accumulation_steps=8,
    output_dir=f"{LOCATION}/model-bart", # 모델 저장 경로
    overwrite_output_dir=True,  # 기존 모델을 덮어쓰기
    learning_rate=lr,  # 학습률 설정
    num_train_epochs=20,  # 학습 에포크 설정
    evaluation_strategy="epoch",  # 평가 스트레티지 설정
    per_device_train_batch_size=16,  # 배치 크기 설정
    per_device_eval_batch_size=16,
    save_steps=1000,  # 모델 저장 스텝 설정
    save_total_limit=10,  # 최대 모델 저장 개수 설정
    warmup_steps=500,  # 워밍업 스텝 설정
    weight_decay=0.01,
    predict_with_generate=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    data_collator=data_collactor,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EmptyCacheCallback()],
)

trainer.train(resume_from_checkpoint = True if get_latest_checkpoint(f'{LOCATION}/model-bart') else False)
trainer.evaluate()

trainer.save_model(f"{LOCATION}/bart")
