import jwt from 'jsonwebtoken'
import { Request, Response, NextFunction } from 'express'

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret'

export const signToken = (sub: string) => jwt.sign({ sub }, JWT_SECRET, { expiresIn: '7d' })

export const requireAuth = (req: Request, res: Response, next: NextFunction) => {
  const h = req.headers.authorization || ''
  const token = h.startsWith('Bearer ') ? h.slice(7) : null
  if (!token) return res.status(401).json({ error: 'missing token' })
  try {
    const payload = jwt.verify(token, JWT_SECRET) as any
    ;(req as any).user = { id: payload.sub }
    next()
  } catch {
    return res.status(401).json({ error: 'invalid token' })
  }
}
