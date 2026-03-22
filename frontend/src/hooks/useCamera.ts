import { useCallback, useRef, useState } from 'react'

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const intervalRef = useRef<number | null>(null)
  const [isActive, setIsActive] = useState(false)

  const start = useCallback(async (onFrame: (blob: Blob) => void) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // Create offscreen canvas for capturing frames
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas')
        canvasRef.current.width = 640
        canvasRef.current.height = 640
      }

      setIsActive(true)

      // Capture frames at ~15fps (balanced for WebSocket latency)
      intervalRef.current = window.setInterval(() => {
        if (!videoRef.current || !canvasRef.current) return
        const ctx = canvasRef.current.getContext('2d')
        if (!ctx) return
        ctx.drawImage(videoRef.current, 0, 0, 640, 640)
        canvasRef.current.toBlob(
          (blob) => { if (blob) onFrame(blob) },
          'image/jpeg',
          0.8,
        )
      }, 66) // ~15fps
    } catch (err) {
      console.error('Camera access failed:', err)
      throw err
    }
  }, [])

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsActive(false)
  }, [])

  return { videoRef, isActive, start, stop }
}
