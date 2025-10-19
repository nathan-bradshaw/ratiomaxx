from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.stream import router as stream_router
from routes.analyze import router as analyze_router

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000', '*'],
  allow_methods=['*'],
  allow_headers=['*']
)

@app.get('/')
def health():
  return {'msg': 'RatioMaxx ML service running'}

app.include_router(stream_router)
app.include_router(analyze_router)