from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from core.ratios import analyze_landmarks

router = APIRouter()

@router.websocket('/ws/analyze')
async def ws_analyze(ws: WebSocket):
  await ws.accept()
  # per-socket EMA state
  state = { 'r_ema': {}, 'r_jit': {} }
  try:
    while True:
      msg = await ws.receive()
      if msg.get('text') is None:
        await ws.send_text(json.dumps({'error': 'expected JSON landmarks'}))
        continue
      try:
        p = json.loads(msg['text'])
      except Exception:
        await ws.send_text(json.dumps({'error': 'invalid json'}))
        continue
      out = analyze_landmarks(p, state)
      await ws.send_text(json.dumps(out))
  except WebSocketDisconnect:
    pass