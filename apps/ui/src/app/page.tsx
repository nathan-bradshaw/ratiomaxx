'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'

export const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState<string | null>('')
  const router = useRouter()

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErr(null)
    try {
      const r = await fetch(`${API}/auth/login`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, password })
      })
      if (!r.ok) throw new Error(`http ${r.status}`)
      const { token } = await r.json()
      localStorage.setItem('rx_token', token)
      router.push('/home')
    } catch (e: any) {
      setErr('login failed')
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6 bg-zinc-900 text-zinc-100">
      <form onSubmit={onSubmit} className="w-full max-w-xs space-y-3">
        <h1 className="text-xl font-semibold">sign in</h1>
        {err && <div className="text-red-500 text-sm">{err}</div>}
        <input
          className="w-full border rounded px-3 py-2"
          placeholder="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          className="w-full border rounded px-3 py-2"
          placeholder="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="w-full bg-black text-white px3 py-2">sign in</button>
        <div className="text-xs opacity-60">
          dev links:{' '}
          <a className="underline" href="/dev">
            /dev
          </a>
        </div>
      </form>
    </main>
  )
}
