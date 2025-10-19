export const startStream = async (
  onResult: (d: any) => void,
  videoEl?: HTMLVideoElement,
  options?: { url?: string; targetFps?: number; jpegQuality?: number; onFrameSent?: () => void; getPolys?: () => any | null }
) => {
  const url = options?.url ?? (process.env.NEXT_PUBLIC_ML_WS_URL || 'ws://localhost:8000/ws/stream')
  const ws = new WebSocket(url)
  await new Promise((r) => (ws.onopen = () => r(null)))

  const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
  const video = videoEl ?? document.createElement('video')
  video.autoplay = true
  video.muted = true
  video.playsInline = true
  video.srcObject = media

  await new Promise((resolve) => {
    if (video.readyState >= 2) return resolve(null)
    video.onloadedmetadata = () => resolve(null)
  })

  try {
    await video.play()
  } catch (e) {
    console.error('video.play() failed', e)
  }

  if (!videoEl && !document.body.contains(video)) {
    document.body.appendChild(video)
  }

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!

  const toArrayBuffer = (canvas: HTMLCanvasElement, quality: number) =>
    new Promise<ArrayBuffer | null>((res) =>
      canvas.toBlob((b) => (b ? b.arrayBuffer().then(res) : res(null)), 'image/jpeg', quality)
    )

  canvas.width = 640
  canvas.height = 480
  console.log('camera stream started', { w: canvas.width, h: canvas.height })

  let fps = options?.targetFps ?? 12
  let q = options?.jpegQuality ?? 0.55
  const FPS_MIN = 6
  const FPS_MAX = 20
  const Q_MIN = 0.4
  const Q_MAX = 0.8
  const BACKPRESSURE_HIGH = 512 * 1024
  const BACKPRESSURE_LOW = 128 * 1024

  let last = 0

  let pausedByVisibility = false
  const onVis = () => {
    pausedByVisibility = document.hidden
  }
  document.addEventListener('visibilitychange', onVis)

  const tick = (t: number) => {
    if (pausedByVisibility) {
      requestAnimationFrame(tick)
      return
    }

    if (t - last > 1000 / fps && ws.readyState === WebSocket.OPEN) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      toArrayBuffer(canvas, q).then((buf) => {
        if (!buf || ws.readyState !== WebSocket.OPEN) return

        const polys = options?.getPolys?.()
        if (polys) ws.send(JSON.stringify(polys))

        ws.send(buf)
        options?.onFrameSent?.()

        const buffered = ws.bufferedAmount
        if (buffered > BACKPRESSURE_HIGH) {
          fps = Math.max(FPS_MIN, fps - 1)
          q = Math.max(Q_MIN, +(q - 0.03).toFixed(2))
        } else if (buffered < BACKPRESSURE_LOW) {
          fps = Math.min(FPS_MAX, fps + 1)
          q = Math.min(Q_MAX, +(q + 0.02).toFixed(2))
        }
      })

      last = t
    }

    requestAnimationFrame(tick)
  }

  ws.onmessage = (ev) => {
    try {
      onResult(JSON.parse(ev.data))
    } catch {
      // ignore invalid JSON
    }
  }
  requestAnimationFrame(tick)

  const stop = () => {
    ws.close()
    media.getTracks().forEach((tr) => tr.stop())
    document.removeEventListener('visibilitychange', onVis)
  }

  return { stop, ws }
}
