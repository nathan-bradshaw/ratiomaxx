export const startStream = async (onResult: (d: any) => void, videoEl?: HTMLVideoElement) => {
  const url = process.env.NEXT_PUBLIC_ML_WS_URL as string
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

  canvas.width = 640
  canvas.height = 480
  console.log('camera stream started', { w: canvas.width, h: canvas.height })

  let last = 0
  const targetFps = 10

  const tick = (t: number) => {
    if (t - last > 1000 / targetFps && ws.readyState === WebSocket.OPEN) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      canvas.toBlob((b) => b && b.arrayBuffer().then((buf) => ws.send(buf)), 'image/jpeg', 0.6)
      last = t
    }
    requestAnimationFrame(tick)
  }

  ws.onmessage = (ev) => onResult(JSON.parse(ev.data))
  requestAnimationFrame(tick)

  return () => {
    ws.close()
    media.getTracks().forEach((tr) => tr.stop())
  }
}
