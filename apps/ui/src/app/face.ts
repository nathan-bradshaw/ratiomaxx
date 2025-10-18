import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision'

let landmarker: FaceLandmarker | null = null

export const loadLandmarker = async () => {
  if (landmarker) return landmarker
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  )
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false
  })
  return landmarker
}

export const drawLandmarks = (
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  mode: 'outline' | 'mesh' = 'outline',
  onLandmarks?: (lm: { x: number; y: number }[]) => void
) => {
  const ctx = canvas.getContext('2d')!
  const utils = new DrawingUtils(ctx)

  const step = async () => {
    if (!landmarker) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const res = landmarker.detectForVideo(video, performance.now())
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (res.faceLandmarks?.length) {
      const lm = res.faceLandmarks[0]

      if (onLandmarks) {
        const now = performance.now()
        ;(drawLandmarks as any)._last = (drawLandmarks as any)._last ?? 0
        if (now - (drawLandmarks as any)._last > 250) {
          onLandmarks(lm.map((p) => ({ x: p.x, y: p.y })))
          ;(drawLandmarks as any)._last = now
        }
      }

      // draw key feature outlines (slightly more noticeable)
      const subtle = { color: 'rgba(255,255,255,0.26)', lineWidth: 0.9 }
      const subtleEye = { color: 'rgba(255,255,255,0.30)', lineWidth: 1.0 }

      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, subtle)
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, subtle)
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, subtleEye)
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, subtleEye)
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, subtle)
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, subtle)
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_NOSE, subtle)

      // optional full mesh overlay (make it extra subtle)
      if (mode === 'mesh') {
        utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
          color: 'rgba(255,255,255,0.12)',
          lineWidth: 0.4
        })
      }

      // simple ratio examples
      const w = canvas.width
      const h = canvas.height
      const p = (i: number) => ({ x: lm[i].x * w, y: lm[i].y * h })
      const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
        Math.hypot(a.x - b.x, a.y - b.y)

      // interpupillary distance vs face width
      const leftEye = p(468) // iris center approx (requires iris subset in model v1; fallback nearby points)
      const rightEye = p(473)
      const eyeDist = dist(leftEye, rightEye)

      // face width: distance between left/right cheek landmarks (rough)
      const leftCheek = p(234)
      const rightCheek = p(454)
      const faceWidth = dist(leftCheek, rightCheek)

      // depth cue: emphasize jawline & cheekbones with brighter depth dots
      ctx.save()
      for (let i = 0; i < lm.length; i++) {
        const pt = lm[i]
        const xy = p(i)
        const z = pt.z ?? 0
        const near = Math.max(0, Math.min(1, -z * 2))
        const isJawOrCheek = (i >= 200 && i <= 234) || i === 234 || i === 454
        const baseAlpha = isJawOrCheek ? 0.18 : 0.08
        const baseRadius = isJawOrCheek ? 1.6 : 1.0
        const r = baseRadius + near * 1.8
        ctx.globalAlpha = baseAlpha + near * 0.18
        ctx.beginPath()
        ctx.arc(xy.x, xy.y, r, 0, Math.PI * 2)
        ctx.fillStyle = isJawOrCheek ? '#00e6c0' : '#f0f0f0'
        ctx.fill()
      }
      ctx.restore()

      const ratioEyeToFace = +(eyeDist / faceWidth).toFixed(3)

      ctx.fillStyle = 'rgba(0,0,0,0.6)'
      ctx.fillRect(8, 8, 180, 50)
      ctx.fillStyle = '#00ff88'
      ctx.font = '12px monospace'
      ctx.fillText(`eye/face: ${ratioEyeToFace}`, 14, 26)
      ctx.fillText(`landmarks: ${lm.length}`, 14, 44)
    }

    requestAnimationFrame(step)
  }

  requestAnimationFrame(step)
}
