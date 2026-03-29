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
  const reconnectTimerRef = useRef<number | null>(null)
  const inFlightRef = useRef(false)
  const [state, setState] = useState<WSState>({
    detections: [],
    inferenceTime: 0,
    frameId: 0,
    isConnected: false,
  })

  const connect = useCallback(function connectWebSocket() {
    if (!token || wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/detect?token=${token}`)
    wsRef.current = ws

    ws.onopen = () => {
      inFlightRef.current = false
      setState((s) => ({ ...s, isConnected: true }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.detections !== undefined) {
          inFlightRef.current = false
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

    ws.onclose = (event) => {
      inFlightRef.current = false
      wsRef.current = null
      setState((s) => ({ ...s, isConnected: false }))
      if (event.code === 4001) {
        localStorage.removeItem('token')
        localStorage.removeItem('user')
        window.location.href = '/login'
        return
      }
      reconnectTimerRef.current = window.setTimeout(() => connectWebSocket(), 2000)
    }

    ws.onerror = () => ws.close()
  }, [token])

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    wsRef.current?.close()
    wsRef.current = null
    inFlightRef.current = false
    setState((s) => ({ ...s, isConnected: false }))
  }, [])

  const sendFrame = useCallback((blob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && !inFlightRef.current) {
      inFlightRef.current = true
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
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
      }
      wsRef.current?.close()
    }
  }, [])

  return { ...state, connect, disconnect, sendFrame, sendConfig }
}
