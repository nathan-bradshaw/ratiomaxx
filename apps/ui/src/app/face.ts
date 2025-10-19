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
  onLandmarks?: (
    lm: { x: number; y: number; z?: number }[],
    meta?: {
      w: number
      h: number
    }
  ) => void
) => {
  const ctx = canvas.getContext('2d')!
  const utils = new DrawingUtils(ctx)

  const step = async () => {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
      requestAnimationFrame(step)
      return
    }
    if (!landmarker) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    let res: any
    try {
      res = landmarker.detectForVideo(video, performance.now())
    } catch {
      requestAnimationFrame(step)
      return
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (res.faceLandmarks?.length) {
      const lm = res.faceLandmarks[0]

      if (onLandmarks) {
        const now = performance.now()
        ;(drawLandmarks as any)._last = (drawLandmarks as any)._last ?? 0
        if (now - (drawLandmarks as any)._last > 250) {
          const w = canvas.width
          const h = canvas.height

          onLandmarks(
            lm.map((p) =>
              'z' in (p as any) ? { x: p.x, y: p.y, z: (p as any).z } : { x: p.x, y: p.y }
            ),
            { w, h }
          )
          ;(drawLandmarks as any)._last = now
        }
      }

      // optional mesh overlay (kept subtle, draws under the outline)
      if (mode === 'mesh') {
        ctx.save()
        ctx.setLineDash([])
        utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
          color: 'rgba(0,255,200,0.25)',
          lineWidth: 0.5
        })
        ctx.restore()
      }

      // light face outline (keeps it subtle)
      ctx.save()
      ctx.setLineDash([3, 3])
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
        color: 'rgba(255,255,255,0.35)',
        lineWidth: 1
      })
      ctx.restore()
    }

    requestAnimationFrame(step)
  }

  requestAnimationFrame(step)
}
