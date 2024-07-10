from fastapi import FastAPI, File, UploadFile 
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image 

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello world!"

def read_file_as_image(data ) -> np.ndarray  :
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    images = read_file_as_image(await file.read())
    return 

@app.post("/predict")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
