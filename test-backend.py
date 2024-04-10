from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# 모델 및 토크나이저 불러오기
model_path = './gpt'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>'
)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()  # 추론 모드로 설정

app = FastAPI()

class TextList(BaseModel):
    texts: List[str]

@app.post("/generate/", response_model=TextList)
async def generate(text_list: TextList):
    try:
        generated_texts = []
        for text in text_list.texts:
            input_ids = tokenizer.encode(text, return_tensors="pt")
            output = model.generate(input_ids, max_length=128)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
        return {"texts": generated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)