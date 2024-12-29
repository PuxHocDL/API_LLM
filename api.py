import io 
import pickle

import numpy as np 
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pydantic import BaseModel
from typing import List, Dict



MODEL_PATH = "lua_model.gguf" 
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)


with open('model_predict.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

class Message(BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str  # Nội dung tin nhắn

class ChatRequest(BaseModel):
    messages: List[Message]  # Danh sách các tin nhắn
    max_tokens: int = 100  # Số token tối đa cho câu trả lời
    temperature: float = 0.7  # Độ sáng tạo của câu trả lời

app = FastAPI()


def create_prompt(messages: List[Message]) -> str:
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "  # Thêm vai trò của chatbot
    return prompt


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil_image = pil_image.resize((28, 28), PIL.Image.LANCZOS)
    im_array = np.array(pil_image).reshape(1,-1) 
    prediction = model.predict(im_array)
    return {"prediction": prediction[0]}

@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    try:
        # Tạo prompt từ chat template
        prompt = create_prompt(chat_request.messages)

        # Gọi mô hình để tạo câu trả lời
        response = llm(
            prompt,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            stop=["user:"],  # Dừng khi gặp tin nhắn từ người dùng
        )

        # Trả về câu trả lời
        return {"response": response["choices"][0]["text"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


