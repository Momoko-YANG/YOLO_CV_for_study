import { useCallback, useEffect, useRef, useState } from 'react'
import Sidebar from '../components/layout/Sidebar'
import TitleBar from '../components/layout/TitleBar'
import DetectionCanvas from '../components/detection/DetectionCanvas'
import ResultsTable from '../components/detection/ResultsTable'
import DetectionInfo from '../components/detection/DetectionInfo'
import ProgressBar from '../components/detection/ProgressBar'
import Slider from '../components/controls/ConfidenceSlider'
import { useCamera } from '../hooks/useCamera'
import { useWebSocket } from '../hooks/useWebSocket'
import { useDetectionStore } from '../hooks/useDetection'
import { useAuthStore } from '../hooks/useAuth'
import api from '../services/api'
import type { ImageDetectionResponse } from '../types/detection'

export default function MainPage() {
  const token = useAuthStore((s) => s.token)
  const camera = useCamera()
  const ws = useWebSocket(token)
  const store = useDetectionStore()
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<number | null>(null)

  // Fetch available models on mount
  useEffect(() => {
    api.get('/api/detect/models').then((res) => {
      setModels(res.data.models)
      setCurrentModel(res.data.current)
    }).catch(() => {})
  }, [])

  // --- Camera ---
  const handleCamera = useCallback(() => {
    if (camera.isActive) {
      camera.stop()
      ws.disconnect()
      store.setMode('idle')
      store.setDetections([], 0)
    } else {
      store.setMode('camera')
      setImageUrl(null)
      ws.connect()
      camera.start((blob) => ws.sendFrame(blob))
    }
  }, [camera, ws, store])

  // Sync WS detections to store
  const prevFrameId = useRef(0)
  if (ws.frameId !== prevFrameId.current) {
    prevFrameId.current = ws.frameId
    store.setDetections(ws.detections, ws.inferenceTime)
  }

  // Send config when sliders change
  const handleConfChange = useCallback((v: number) => {
    store.setConf(v)
    ws.sendConfig(v, store.iou)
  }, [ws, store])

  const handleIouChange = useCallback((v: number) => {
    store.setIou(v)
    ws.sendConfig(store.conf, v)
  }, [ws, store])

  // --- Image upload ---
  const handleImage = useCallback(() => {
    imageInputRef.current?.click()
  }, [])

  const onImageSelected = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    camera.stop()
    ws.disconnect()
    store.setMode('image')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('conf', String(store.conf))
    formData.append('iou', String(store.iou))

    try {
      const res = await api.post<ImageDetectionResponse>('/api/detect/image', formData)
      setImageUrl(res.data.image_url)
      store.setDetections(res.data.detections, res.data.inference_time)
    } catch (err) {
      console.error('Image detection failed:', err)
    }
    e.target.value = ''
  }, [camera, ws, store])

  // --- Video upload ---
  const handleVideo = useCallback(() => {
    videoInputRef.current?.click()
  }, [])

  const onVideoSelected = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    camera.stop()
    ws.disconnect()
    store.setMode('video')
    store.setVideoProgress(0)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('conf', String(store.conf))
    formData.append('iou', String(store.iou))

    try {
      const res = await api.post('/api/detect/video', formData)
      const taskId = res.data.task_id
      store.setVideoTaskId(taskId)

      // Poll for progress
      pollRef.current = window.setInterval(async () => {
        const status = await api.get(`/api/detect/video/${taskId}/status`)
        store.setVideoProgress(status.data.progress)
        if (status.data.status === 'completed') {
          if (pollRef.current) clearInterval(pollRef.current)
          const results = await api.get(`/api/detect/video/${taskId}/results`)
          // Show last frame's detections
          const frames = results.data.frames
          if (frames.length > 0) {
            store.setDetections(frames[frames.length - 1], 0)
          }
        }
      }, 500)
    } catch (err) {
      console.error('Video upload failed:', err)
    }
    e.target.value = ''
  }, [camera, ws, store])

  // --- Folder ---
  const handleFolder = useCallback(() => {
    folderInputRef.current?.click()
  }, [])

  const onFolderSelected = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return
    camera.stop()
    ws.disconnect()
    store.setMode('image')

    // Process first image
    const formData = new FormData()
    formData.append('file', files[0])
    formData.append('conf', String(store.conf))
    formData.append('iou', String(store.iou))

    try {
      const res = await api.post<ImageDetectionResponse>('/api/detect/image', formData)
      setImageUrl(res.data.image_url)
      store.setDetections(res.data.detections, res.data.inference_time)
    } catch (err) {
      console.error('Image detection failed:', err)
    }
    e.target.value = ''
  }, [camera, ws, store])

  // --- Model switch ---
  const handleModelSwitch = useCallback(async (model: string) => {
    try {
      await api.post('/api/detect/model/switch', { model })
      setCurrentModel(model)
    } catch (err) {
      console.error('Model switch failed:', err)
    }
  }, [])

  // --- Save/Export ---
  const handleSave = useCallback(async () => {
    if (store.videoTaskId) {
      try {
        const res = await api.get(`/api/detect/export/csv/${store.videoTaskId}`, { responseType: 'blob' })
        const url = URL.createObjectURL(res.data)
        const a = document.createElement('a')
        a.href = url
        a.download = 'results.csv'
        a.click()
        URL.revokeObjectURL(url)
      } catch (err) {
        console.error('Export failed:', err)
      }
    }
  }, [store.videoTaskId])

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-50 to-emerald-50/30">
      <TitleBar onSave={handleSave} />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          onCamera={handleCamera}
          onImage={handleImage}
          onVideo={handleVideo}
          onFolder={handleFolder}
          models={models}
          currentModel={currentModel}
          onModelSwitch={handleModelSwitch}
          mode={store.mode}
        />

        {/* Main content */}
        <div className="flex-1 flex gap-4 p-4 overflow-hidden">
          {/* Left: Canvas + controls */}
          <div className="flex-1 flex flex-col gap-3 min-w-0">
            {/* Sliders */}
            <div className="flex gap-4">
              <div className="flex-1">
                <Slider label="Conf" value={store.conf} onChange={handleConfChange} />
              </div>
              <div className="flex-1">
                <Slider label="IoU" value={store.iou} onChange={handleIouChange} />
              </div>
            </div>

            {/* Detection canvas */}
            <div className="flex-1 flex items-center justify-center">
              <DetectionCanvas
                videoRef={camera.videoRef}
                imageUrl={imageUrl}
                detections={store.detections}
                width={640}
                height={480}
              />
            </div>

            {/* Progress bar */}
            <ProgressBar progress={store.videoProgress} visible={store.mode === 'video'} />

            {/* Results table */}
            <div className="max-h-48 overflow-y-auto">
              <ResultsTable
                detections={store.detections}
                selectedIndex={store.selectedIndex}
                onSelect={(i) => store.setSelectedIndex(i)}
              />
            </div>
          </div>

          {/* Right: Info panel */}
          <div className="w-64 flex-shrink-0">
            <DetectionInfo
              detections={store.detections}
              inferenceTime={store.inferenceTime}
              selectedIndex={store.selectedIndex}
              onSelectTarget={(i) => store.setSelectedIndex(i)}
            />
          </div>
        </div>
      </div>

      {/* Hidden file inputs */}
      <input ref={imageInputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden" onChange={onImageSelected} />
      <input ref={videoInputRef} type="file" accept=".mp4,.avi" className="hidden" onChange={onVideoSelected} />
      <input ref={folderInputRef} type="file" accept=".jpg,.jpeg,.png" multiple className="hidden" onChange={onFolderSelected} />

      {/* Hidden video element for camera */}
      <video ref={camera.videoRef} className="hidden" playsInline muted />
    </div>
  )
}
