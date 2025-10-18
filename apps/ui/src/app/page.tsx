'use client'

import { useState, useRef } from 'react'
import { startStream } from './stream'
import { loadLandmarker, drawLandmarks } from './face'

export const ping = async () => fetch('http://localhost:4000/ping').then((r) => r.json())

export default function Home() {
  const [pingResult, setPingResult] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const [running, setRunning] = useState(false)
  const [fps, setFps] = useState(0)
  const [overlayMode, setOverlayMode] = useState<'outline' | 'mesh'>('outline')
  const lastRef = useRef({ t: 0, count: 0 })
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const lastLmRef = useRef<any[] | null>(null)

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
  }

  const onStart = async () => {
    if (running) return
    await loadLandmarker()
    const stop = await startStream(setResult, videoRef.current ?? undefined)
    ;(window as any).stopStream = stop
    setRunning(true)
    if (videoRef.current && overlayRef.current)
      drawLandmarks(videoRef.current, overlayRef.current, overlayMode, onLm)
  }

  const onStop = () => {
    if (!running) return
    ;(window as any).stopStream?.()
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

  const handleAnalyzeLandmarks = async () => {
    try {
      const lm = lastLmRef.current
      if (!lm || lm.length < 474) {
        setPingResult('No live landmarks yet — start the stream and ensure your face is visible')
        return
      }
      const base = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000'
      const w = videoRef.current?.videoWidth ?? 0
      const h = videoRef.current?.videoHeight ?? 0
      const sample = { landmarks: lm, width: w, height: h }

      const res = await fetch(`${base}/analyze/landmarks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sample)
      })

      if (!res.ok) {
        const txt = await res.text()
        setPingResult(`error ${res.status}: ${txt}`)
        return
      }

      const data = await res.json()
      setPingResult(JSON.stringify(data))
    } catch (err) {
      setPingResult('Failed to analyze landmarks')
    }
  }

  return (
    <main className="p-4">
      <div className="mb-3">
        <div className="relative inline-block">
          <video
            ref={videoRef}
            width={640}
            height={480}
            className="rounded border"
            autoPlay
            muted
            playsInline
          />
          <canvas ref={overlayRef} className="absolute inset-0 pointer-events-none" />
        </div>
        <p className="text-xs mt-1">fps: {fps}</p>
      </div>
      <button onClick={handlePing} className="border px-4 py-2">
        Ping Server
      </button>
      <button onClick={handleAnalyzeLandmarks} className="border px-4 py-2 ml-2">
        Analyze Landmarks
      </button>
      {pingResult && <p className="mt-2">{pingResult}</p>}
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xs">overlay:</span>
        <button
          className={`border px-2 py-1 text-xs ${overlayMode === 'outline' ? 'bg-gray-100' : ''}`}
          onClick={() => setOverlayMode('outline')}
        >
          outline
        </button>
        <button
          className={`border px-2 py-1 text-xs ${overlayMode === 'mesh' ? 'bg-gray-100' : ''}`}
          onClick={() => setOverlayMode('mesh')}
        >
          mesh
        </button>
      </div>
      <div className="flex gap-2">
        <button className="border px-3 py-2" onClick={onStart}>
          start
        </button>
        <button className="border px-3 py-2" onClick={onStop}>
          stop
        </button>
      </div>
      {result && (
        <pre className="text-xs bg-gray-100 p-3 rounded w-full max-w-xl overflow-auto mt-3">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  )
}
