'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

export default function HomePage() {
  const [ready, setReady] = useState(false)
  const router = useRouter()

  useEffect(() => {
    const t = localStorage.getItem('rx_token')
    if (!t) router.replace('/')
    else setReady(true)
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
