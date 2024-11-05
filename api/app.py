from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

# Load model and tokenizer
model_name = "llama/Llama3.2-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

class InputData(BaseModel):
    input_text: str

def predict_sbp_dbp(input_text):
    inputs = tokenizer(input_text, return_tensor="pt")
    with torch.no_grad():
        output = model(**inputs)
    return output.logits.tolist()

@app.post("/predict")
async def predict(input_text: InputData):
    result = predict_sbp_dbp(input_text)
    return {"Predicted_values": result}