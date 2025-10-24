import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import morgan from 'morgan'
import { signToken, requireAuth, AuthenticatedRequest } from './auth.js'
import type { Request, Response } from 'express'

dotenv.config()
const app = express()
const port = process.env.PORT || 4000

type LoginRequestBody = { email?: string; password?: string }

app.use(cors({ origin: process.env.UI_ORIGIN }))
app.use(express.json())
app.use(morgan('tiny'))

// health check
app.get('/', (req, res) => res.json({ msg: 'RatioMaxx API running' }))

// ping test for UI
app.get('/ping', (req, res) => res.json({ ok: true, msg: 'pong from API' }))

// upload endpoint (stub for future ML service)
app.post('/analyze', (req, res) => {
  // placeholder until Python service exists
  res.json({
    status: 'received',
    message: 'Facial analysis pending implementation'
  })
})

app.post('/auth/login', (req: Request, res: Response) => {
  const { email, password } = (req.body || {}) as LoginRequestBody
  if (!email || !password) {
    res.status(400).json({ error: 'bad creds' })
    return
  }
  const token = signToken(email)
  res.json({ token })
})

app.get('/auth/me', requireAuth, (req: AuthenticatedRequest, res: Response) => {
  res.json({ user: req.user })
})

app.listen(port, () => console.log(`API listening on http://localhost:${port}`))
