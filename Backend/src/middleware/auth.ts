import { Request, Response, NextFunction } from 'express'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

export interface AuthenticatedRequest extends Request {
  userId?: string
  user?: any
}

export async function authenticateUser(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  try {
    const authHeader = req.headers.authorization
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No authorization token provided' })
    }

    const token = authHeader.substring(7)
    const { data: { user }, error } = await supabase.auth.getUser(token)

    if (error || !user) {
      return res.status(401).json({ error: 'Invalid or expired token' })
    }

    req.userId = user.id
    req.user = user
    next()
  } catch (error) {
    console.error('Auth middleware error:', error)
    return res.status(500).json({ error: 'Authentication failed' })
  }
}

// Optional auth middleware - doesn't fail if no token provided
export async function optionalAuth(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  try {
    const authHeader = req.headers.authorization
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const token = authHeader.substring(7)
      const { data: { user }, error } = await supabase.auth.getUser(token)
      
      if (!error && user) {
        req.userId = user.id
        req.user = user
      }
    }
    next()
  } catch (error) {
    console.error('Optional auth middleware error:', error)
    next()
  }
}
