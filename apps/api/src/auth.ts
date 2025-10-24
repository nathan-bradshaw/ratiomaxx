import jwt from 'jsonwebtoken'
import type { Request, Response, NextFunction, RequestHandler } from 'express'

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret'

type JwtPayload = { email: string }

export type AuthenticatedRequest = Request & { user?: { email: string } }

export const signToken = (email: string) => jwt.sign({ email }, JWT_SECRET, { expiresIn: '1h' })

export const requireAuth: RequestHandler = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
) => {
  const h = req.headers.authorization || ''
  const token = h.startsWith('Bearer ') ? h.slice(7) : null
  if (!token) {
    res.status(401).json({ error: 'missing token' })
    return
  }
  try {
    const payload = jwt.verify(token, JWT_SECRET) as JwtPayload
    req.user = { email: payload.email }
    next()
  } catch {
    res.status(401).json({ error: 'invalid token' })
  }
}
