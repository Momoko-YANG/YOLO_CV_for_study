import { useCallback, useEffect, useRef, useState } from 'react'
import type { DetectionResult } from '../types/detection'

interface WSState {
  detections: DetectionResult[]
  inferenceTime: number
  frameId: number
  isConnected: boolean
}

export function useWebSocket(token: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const [state, setState] = useState<WSState>({
    detections: [],
    inferenceTime: 0,
    frameId: 0,
    isConnected: false,
  })

  const connect = useCallback(() => {
    if (!token || wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/detect?token=${token}`)
    wsRef.current = ws

    ws.onopen = () => setState((s) => ({ ...s, isConnected: true }))

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.detections !== undefined) {
          setState({
            detections: data.detections,
            inferenceTime: data.inference_time,
            frameId: data.frame_id,
            isConnected: true,
          })
        }
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      setState((s) => ({ ...s, isConnected: false }))
      // Auto-reconnect after 2s
      setTimeout(() => connect(), 2000)
    }

    ws.onerror = () => ws.close()
  }, [token])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
    setState((s) => ({ ...s, isConnected: false }))
  }, [])

  const sendFrame = useCallback((blob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buf) => wsRef.current?.send(buf))
    }
  }, [])

  const sendConfig = useCallback((conf: number, iou: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'config', conf, iou }))
    }
  }, [])

  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  return { ...state, connect, disconnect, sendFrame, sendConfig }
}
