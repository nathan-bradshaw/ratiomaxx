import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'

dotenv.config()
const app = express()
const port = process.env.PORT || 4000

app.use(cors())
app.use(express.json())

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

app.listen(port, () => console.log(`API listening on http://localhost:${port}`))
