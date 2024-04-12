from torch.utils.data import Dataset, dataloader
from transformers import AutoTokenizer

import pandas as pd
import torch

LOCATION = '.'

# GPT-3 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
max_length = 128

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token = '[CLS]'
        self.eos_token = '[SEP]'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data.iloc[idx]['text'])
        corrected_text = str(self.data.iloc[idx]['corrected'])

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

train = pd.read_csv(f'{LOCATION}/train.csv')
validate = pd.read_csv(f'{LOCATION}/validate.csv')

train_dataset = TextDataset(train, tokenizer, max_length)
validate_dataset = TextDataset(validate, tokenizer, max_length)

from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
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

model = AutoModelForMaskedLM.from_pretrained('monologg/distilkobert')
lr=2e-5
SAVE_PATH = f"{LOCATION}/all_metrics.pkl"

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "rb") as f:
        all_metrics = pickle.load(f)
else:
    all_metrics = []

def get_latest_checkpoint(path):
    checkpoints = [f for f in os.listdir(path) if f.startswith('checkpoint-')]
    if checkpoints:
        return os.path.join(path, max(checkpoints))
    else:
        return None

def compute_metrics(eval_pred):
    metric_bleu = load_metric("bleu")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = metric_bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    return {"bleu": bleu_score["score"]}

class EmptyCacheCallback(TrainerCallback):
    """훈련의 각 로그 스텝마다 CUDA 캐시를 비우는 콜백"""

    def on_log(self, args, state, control, **kwargs):
        """로그 이벤트가 발생할 때 CUDA 캐시를 비웁니다."""
        gc.collect()
        torch.mps.empty_cache()

data_collactor = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=f"{LOCATION}/model", # 모델 저장 경로
    overwrite_output_dir=True,  # 기존 모델을 덮어쓰기
    learning_rate=lr,  # 학습률 설정
    num_train_epochs=20,  # 학습 에포크 설정
    evaluation_strategy="epoch",  # 평가 스트레티지 설정
    per_device_train_batch_size=16,  # 배치 크기 설정
    per_device_eval_batch_size=1,
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
    compute_metrics=compute_metrics,
    callbacks=[EmptyCacheCallback()]
)

trainer.train(resume_from_checkpoint = True if get_latest_checkpoint(f'{LOCATION}/bart/') else False)
trainer.evaluate()

trainer.save_model(f"{LOCATION}/gpt2")