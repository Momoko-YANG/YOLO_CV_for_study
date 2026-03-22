import { useEffect, useRef } from 'react'
import type { DetectionResult } from '../../types/detection'
import { drawDetections } from '../../utils/canvas'

interface DetectionCanvasProps {
  videoRef?: React.RefObject<HTMLVideoElement | null>
  imageUrl?: string | null
  detections: DetectionResult[]
  width?: number
  height?: number
}

export default function DetectionCanvas({
  videoRef,
  imageUrl,
  detections,
  width = 640,
  height = 480,
}: DetectionCanvasProps) {
  const bgCanvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)

  // Draw video frames continuously
  useEffect(() => {
    if (!videoRef?.current) return

    const draw = () => {
      const bgCtx = bgCanvasRef.current?.getContext('2d')
      if (bgCtx && videoRef.current && videoRef.current.readyState >= 2) {
        bgCtx.drawImage(videoRef.current, 0, 0, width, height)
      }
      animRef.current = requestAnimationFrame(draw)
    }
    animRef.current = requestAnimationFrame(draw)

    return () => cancelAnimationFrame(animRef.current)
  }, [videoRef, width, height])

  // Draw image when imageUrl changes
  useEffect(() => {
    if (!imageUrl || !bgCanvasRef.current) return
    const ctx = bgCanvasRef.current.getContext('2d')
    if (!ctx) return
    const img = new Image()
    img.onload = () => ctx.drawImage(img, 0, 0, width, height)
    img.src = imageUrl
  }, [imageUrl, width, height])

  // Draw detection overlays
  useEffect(() => {
    const ctx = overlayCanvasRef.current?.getContext('2d')
    if (!ctx) return
    const scaleX = width / 640
    const scaleY = height / 640
    drawDetections(ctx, detections, scaleX, scaleY)
  }, [detections, width, height])

  return (
    <div className="relative" style={{ width, height }}>
      <canvas
        ref={bgCanvasRef}
        width={width}
        height={height}
        className="absolute inset-0 rounded-lg"
        style={{ background: '#e2e8f0' }}
      />
      <canvas
        ref={overlayCanvasRef}
        width={width}
        height={height}
        className="absolute inset-0 rounded-lg"
      />
    </div>
  )
}
