from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/")
def health():
    return {"msg": "RatioMaxx ML service running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # placeholder: later we’ll use DeepFace / OpenCV here
    contents = await file.read()
    size_kb = round(len(contents) / 1024, 2)
    return {"status": "ok", "file_size_kb": size_kb}