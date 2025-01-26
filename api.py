import io
import pickle
import numpy as np
import PIL.Image
import torch
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms
import base64

# Khởi tạo mô hình ResNet-152
model = models.resnet152(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Load checkpoint
checkpoint_dir = "resnet152.pt"
checkpoint = torch.load(checkpoint_dir, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = [
    'bệnh cháy lá vi khuẩn', 'bệnh đốm nâu', 'lúa khỏe mạnh', 'bệnh đạo ôn lá', 'bệnh cháy lá',
    'bệnh đốm nâu hẹp', 'bệnh đạo ôn cổ bông', 'bệnh bọ gai', 'bệnh khô vằn', 'bệnh tungro'
]


MODEL_PATH = "lua_model.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)


app = FastAPI()

class Message(BaseModel):
    role: str
    content: str
    imageData: Optional[str] = None  

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 0.8

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


def create_disease_prompt(disease_name: str) -> str:
    return f"""
    Bệnh lúa được phát hiện: {disease_name}
    Hãy tư vấn chi tiết các bước để:
    1. Nhận biết triệu chứng của bệnh
    2. Các biện pháp phòng ngừa
    3. Cách điều trị bệnh
    4. Các lưu ý quan trọng khi điều trị
    """

def create_prompt(messages: List[Message], prediction: Optional[str] = None) -> str:
    prompt = ""
    for msg in messages:
        if prediction and msg == messages[-1]:
            prompt += f"user: Kết quả phân tích ảnh: {prediction}\n{msg.content}\n"
        else:
            prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "
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
    try:
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            "prediction": class_names[predicted.item()],
            "confidence": float(confidence.item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    try:
        last_message = chat_request.messages[-1]
        prompt = ""

        if hasattr(last_message, 'imageData') and last_message.imageData:
            image_data = base64.b64decode(last_message.imageData.split(',')[1])
            prediction_result = await predict(UploadFile(
                file=io.BytesIO(image_data),
                filename="image.jpg"
            ))
            
            if last_message.content.strip():
                prompt = create_prompt(chat_request.messages, prediction_result["prediction"])
            else:
                prompt = create_disease_prompt(prediction_result["prediction"])
        else:
            prompt = create_prompt(chat_request.messages)

        response = llm(
            prompt,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            stop=["user:", "assistant:"]
        )

        return {"response": response["choices"][0]["text"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))