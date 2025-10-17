'use client'

import { useState } from 'react'

export const ping = async () => fetch('http://localhost:4000/ping').then((r) => r.json())

export default function Home() {
  const [pingResult, setPingResult] = useState<string | null>(null)

  const handlePing = async () => {
    try {
      const response = await ping()
      console.log(response)
      setPingResult(response.message)
    } catch {
      setPingResult('Failed to fetch ping')
    }
  }

  return (
    <main className="p-4">
      <button onClick={handlePing} className="border px-4 py-2">
        Ping Server
      </button>
      {pingResult && <p className="mt-2">{pingResult}</p>}
    </main>
  )
}
