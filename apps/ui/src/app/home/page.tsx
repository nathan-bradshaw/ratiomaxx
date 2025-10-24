'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { API } from '../page'

export default function HomePage() {
  const [ready, setReady] = useState(false)
  const [user, setUser] = useState('')
  const router = useRouter()

  const checkAuth = async () => {
    const token = localStorage.getItem('rx_token')
    if (!token) {
      router.push(`/`)
      return
    }
    try {
      const res = await fetch(`${API}/auth/me`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!res.ok) {
        router.push(`/`)
        return
      }
      const data = await res.json()
      console.log('user:', data.user)
      setReady(true)
    } catch (e) {
      console.error(`auth check failed `, e)
      router.replace(`/`)
    }
  }

  useEffect(() => {
    checkAuth()
  }, [router])

  if (!ready) return null

  const signOut = () => {
    localStorage.removeItem('rx_token')
    router.replace('/')
  }

  return (
    <main className="py-6 space-y-4">
      <h1 className="text-2xl font-semibold">home</h1>
      <div className="text-sm opacity-70">hi</div>
      <button className="border rounded px-3 py-2" onClick={signOut}>
        sign out
      </button>
    </main>
  )
}
