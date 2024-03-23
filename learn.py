from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, GPT2LMHeadModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

import torch
import pickle
import pandas as pd
import torch

LOCATION = '.'

# GPT-3 모델과 토크나이저 불러오기
tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
max_length = 100
loader_size = 16

# 훈련용 및 검증용 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizerFast, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = str(self.data.iloc[idx]['text'])
        output_text = str(self.data.iloc[idx]['corrected'])
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding='max_length')
        output_ids = self.tokenizer.encode(output_text, max_length=self.max_length, truncation=True, padding='max_length')
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }

train = pd.read_csv(f'{LOCATION}/train.csv')
validate = pd.read_csv(f'{LOCATION}/validate.csv')

train_dataset = TextDataset(train, tokenizer, max_length)
validate_dataset = TextDataset(validate, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True)

device = torch.device('cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

# Model and tokenizer (assuming you have them defined elsewhere)
model = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2").to(device=device)

all_metrics = []

model.train()

lr = 2e-5
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 학습 및 검증 함수 정의
def train(train_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader):
        inputs = batch['input_ids'].to(model.device)
        labels = batch['output_ids'].to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(validate_loader, model):
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(validate_loader):
            inputs = batch['input_ids'].to(model.device)
            labels = batch['output_ids'].to(model.device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            predictions.extend(outputs.logits.argmax(dim=-1).tolist())
            targets.extend(labels.tolist())
    roc_auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')
    return total_loss / len(validate_loader), roc_auc, f1

# 학습 및 검증
epochs = 10
for epoch in tqdm(range(epochs)):
    train_loss = train(train_loader, model, optimizer)
    validate_loss, roc_auc, f1 = validate(validate_loader, model)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Validate Loss: {validate_loss}, ROC-AUC: {roc_auc}, F1 Score: {f1}")
    all_metrics.append({
        'epoch':epoch + 1,
        'train_loss': train_loss,
        'val_loss': validate_loss,
        'roc_auc': roc_auc,
        'f1': f1
    })
    with open("{LOCATION}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)