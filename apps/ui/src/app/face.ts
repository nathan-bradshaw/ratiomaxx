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
  ) => void,
  debugPolys?: boolean
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

      if (debugPolys) {
        const { eyePoly, teethPoly } = buildEyeTeethPolys(lm)
        drawPoly(ctx, eyePoly.left, 'rgba(0,0,0,0.5)', { width: 1, dash: [3, 3] })
        drawPoly(ctx, eyePoly.right, 'rgba(0,0,0,0.5)', { width: 1, dash: [3, 3] })
        drawPoly(ctx, teethPoly, 'rgba(0,0,0,0.5)', { width: 1, dash: [3, 3] })
      }

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

      // light face outline with sharper joins so jaw corners read better
      ctx.save()
      ctx.setLineDash([3, 3])
      ctx.lineJoin = 'miter'
      ctx.miterLimit = 2
      utils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
        color: 'rgba(255,255,255,0.3)',
        lineWidth: 1
      })
      ctx.restore()
    }

    requestAnimationFrame(step)
  }

  requestAnimationFrame(step)
}

function drawPoly(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  color: string,
  opts?: { width?: number; dash?: number[] }
) {
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = opts?.width ?? 1
  ctx.setLineDash(opts?.dash ?? [4, 4])
  ctx.lineJoin = 'miter'
  ctx.miterLimit = 2
  ctx.beginPath()
  if (points.length > 0) {
    ctx.moveTo(points[0].x * ctx.canvas.width, points[0].y * ctx.canvas.height)
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x * ctx.canvas.width, points[i].y * ctx.canvas.height)
    }
    ctx.closePath()
  }
  ctx.stroke()
  ctx.restore()
}

function buildEyeTeethPolys(lms: { x: number; y: number }[]) {
  const pick = (idxs: number[]) => idxs.map((i) => ({ x: lms[i].x, y: lms[i].y }))
  const eyePoly = { left: pick(eyeIdxL), right: pick(eyeIdxR) }
  const teethPoly = pick(innerLipIdx)
  return { eyePoly, teethPoly }
}

// full eye rings (upper + lower) and inner lip ring
const eyeIdxL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
const eyeIdxR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
const innerLipIdx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

export const buildPolys = (lms: { x: number; y: number }[]) => {
  const pick = (idxs: number[]) => idxs.map((i) => ({ x: lms[i].x, y: lms[i].y }))
  const eyePoly = { left: pick(eyeIdxL), right: pick(eyeIdxR) }
  const teethPoly = pick(innerLipIdx)
  return { eyePoly, teethPoly }
}
