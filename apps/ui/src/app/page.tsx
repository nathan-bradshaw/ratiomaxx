'use client'

import { useState, useRef, useEffect } from 'react'
import { startStream } from './stream'
import { loadLandmarker, drawLandmarks } from './face'

export const ping = async () => fetch('http://localhost:4000/ping').then((r) => r.json())

export default function Home() {
  const [pingResult, setPingResult] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [data, setData] = useState<any>(null)

  // pick avg or now
  const pick = (m?: { now?: number; avg?: number }) => m?.avg ?? m?.now

  const [running, setRunning] = useState(false)
  const [fps, setFps] = useState(0)
  const [streamFps, setStreamFps] = useState(0)
  const streamRef = useRef({ t: 0, count: 0 })
  const [overlayMode, setOverlayMode] = useState<'outline' | 'mesh'>('outline')
  const lastRef = useRef({ t: 0, count: 0 })
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const lastLmRef = useRef<any[] | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const analyzeWsRef = useRef<WebSocket | null>(null)

  const [dark, setDark] = useState(false)
  const toggleDark = () => setDark(!dark)

  const openAnalyzeWs = () => {
    const url = process.env.NEXT_PUBLIC_ML_WS_ANALYZE_URL || 'ws://localhost:8000/ws/analyze'
    const ws = new WebSocket(url)
    analyzeWsRef.current = ws
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg) setData(msg) //.data
      } catch {}
    }
    ws.onclose = () => {
      analyzeWsRef.current = null
    }
    ws.onerror = () => {
      try {
        ws.close()
      } catch {}
    }
  }

  const onLm = (lm: any[]) => {
    lastLmRef.current = lm
    const now = performance.now()
    const { t, count } = lastRef.current
    if (now - t > 1000) {
      setFps(count)
      lastRef.current = { t: now, count: 1 }
    } else {
      lastRef.current.count = count + 1
    }
    const w = videoRef.current?.videoWidth ?? 0
    const h = videoRef.current?.videoHeight ?? 0
    const ws = analyzeWsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ landmarks: lm, video: { w, h }, fps }))
    }
  }

  const onStart = async () => {
    if (running) return
    await loadLandmarker()
    const { stop, ws } = await startStream(setResult, videoRef.current ?? undefined, {
      url: process.env.NEXT_PUBLIC_ML_WS_URL || 'ws://localhost:8000/ws/stream',
      onFrameSent: () => {
        const now = performance.now()
        const { t, count } = streamRef.current
        if (now - t > 1000) {
          setStreamFps(count)
          streamRef.current = { t: now, count: 1 }
        } else {
          streamRef.current.count = count + 1
        }
      }
    })
    wsRef.current = ws
    openAnalyzeWs()
    ;(window as any).stopStream = stop
    setRunning(true)
    if (videoRef.current && overlayRef.current)
      drawLandmarks(videoRef.current, overlayRef.current, overlayMode, onLm)
  }

  const onStop = () => {
    if (!running) return
    ;(window as any).stopStream?.()
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close()
    }
    wsRef.current = null
    if (analyzeWsRef.current && analyzeWsRef.current.readyState === WebSocket.OPEN) {
      analyzeWsRef.current.close()
    }
    analyzeWsRef.current = null
    setRunning(false)
  }

  const handlePing = async () => {
    try {
      const response = await ping()
      console.log(response)
      setPingResult(response.msg)
    } catch {
      setPingResult('Failed to fetch ping')
    }
  }

  return (
    <main
      className={`${dark ? 'bg-zinc-900 text-zinc-100' : 'bg-white text-black'} p-4 min-h-screen`}
    >
      <div className="flex items-start gap-4 mb-3">
        <div className="relative inline-block">
          <video
            ref={videoRef}
            width={640}
            height={480}
            className={`rounded border ${dark ? 'border-zinc-700' : 'border-gray-300'}`}
            autoPlay
            muted
            playsInline
          />
          <canvas ref={overlayRef} className="absolute inset-0 pointer-events-none" />
          {result?.skin && (
            <div className="absolute left-2 top-2 bg-black/70 text-white text-xs rounded px-2 py-1 shadow space-y-0.5">
              <div>skin clarity {pick(result.skin.clarity)?.toFixed(1)}</div>
              <div>redness {pick(result.skin.redness)?.toFixed(1)}</div>
              <div>evenness {pick(result.skin.evenness)?.toFixed(1)}</div>
              <div>shine {pick(result.skin.shine)?.toFixed(1)}</div>
              <div>dark circles {pick(result.skin.dark_circles)?.toFixed(1)}</div>
              <div>blemishes {pick(result.skin.blemishes)?.toFixed(1)}</div>
              {result.signals?.teeth_whiteness && (
                <div>
                  teeth {pick(result.signals.teeth_whiteness)?.toFixed(1)}
                  {typeof result.signals.teeth_whiteness.conf === 'number'
                    ? ` (${Math.round(result.signals.teeth_whiteness.conf)}%)`
                    : ''}
                </div>
              )}
              {result.signals?.sclera_health && (
                <div>
                  eye sclera {pick(result.signals.sclera_health)?.toFixed(1)}
                  {typeof result.signals.sclera_health.conf === 'number'
                    ? ` (${Math.round(result.signals.sclera_health.conf)}%)`
                    : ''}
                </div>
              )}
            </div>
          )}
          {data && (
            <div className="absolute top-2 right-2 bg-black/70 text-yellow-300 text-xs rounded px-2 py-1 shadow">
              <div className="font-semibold">
                φ{' '}
                {typeof data.golden.score_pct === 'number' ? data.golden.score_pct.toFixed(1) : '-'}
                %
              </div>
              {data.ratios && (
                <div className="mt-1 text-[10px] text-white/90">
                  <div>face {data.golden.ratios.face_phi}</div>
                  <div>eyes {data.golden.ratios.eyes_phi}</div>
                  <div>mouth/nose {data.golden.ratios.mouth_nose_phi}</div>
                  <div>lip fullness {data.golden.ratios.lip_fullness_phi}</div>
                </div>
              )}
            </div>
          )}
          <p className="text-xs mt-1">
            stream fps: {streamFps} • landmarks fps: {fps}
          </p>
        </div>
        <div className="w-full max-w-xl h-[500px] overflow-auto">
          <pre
            className={`text-xs p-3 rounded border min-h-full ${
              dark
                ? 'bg-zinc-800 text-zinc-100 border-zinc-700'
                : 'bg-gray-100 text-black border-gray-300'
            }`}
          >
            {result || data ? JSON.stringify({ data, result }, null, 2) : 'waiting for data…'}
          </pre>
        </div>
      </div>
      <button
        onClick={handlePing}
        className={`border px-4 py-2 ${
          dark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-gray-300'
        }`}
      >
        Ping Server
      </button>
      {pingResult && <p className="mt-2">{pingResult}</p>}
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xs">overlay:</span>
        <button
          className={`border px-2 py-1 text-xs ${dark ? 'border-zinc-700' : 'border-gray-300'} ${
            overlayMode === 'outline' ? (dark ? 'bg-zinc-800' : 'bg-gray-100') : ''
          }`}
          onClick={() => setOverlayMode('outline')}
        >
          outline
        </button>
        <button
          className={`border px-2 py-1 text-xs ${dark ? 'border-zinc-700' : 'border-gray-300'} ${
            overlayMode === 'mesh' ? (dark ? 'bg-zinc-800' : 'bg-gray-100') : ''
          }`}
          onClick={() => setOverlayMode('mesh')}
        >
          mesh
        </button>
      </div>
      <div className="flex gap-2">
        <button
          className={`border px-3 py-2 ${
            dark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-gray-300'
          }`}
          onClick={onStart}
        >
          start
        </button>
        <button
          className={`border px-3 py-2 ${
            dark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-gray-300'
          }`}
          onClick={onStop}
        >
          stop
        </button>
        <button
          onClick={toggleDark}
          className={`border px-3 py-1 text-xs rounded ${
            dark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-gray-300'
          }`}
        >
          theme: {dark ? 'dark' : 'light'}
        </button>
      </div>
    </main>
  )
}
