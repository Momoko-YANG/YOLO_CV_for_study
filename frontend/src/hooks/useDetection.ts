import { create } from 'zustand'
import type { DetectionResult } from '../types/detection'

interface DetectionState {
  detections: DetectionResult[]
  inferenceTime: number
  selectedIndex: number | null
  conf: number
  iou: number
  mode: 'idle' | 'camera' | 'image' | 'video'
  videoProgress: number
  videoTaskId: string | null
  setDetections: (d: DetectionResult[], time: number) => void
  setSelectedIndex: (i: number | null) => void
  setConf: (v: number) => void
  setIou: (v: number) => void
  setMode: (m: 'idle' | 'camera' | 'image' | 'video') => void
  setVideoProgress: (p: number) => void
  setVideoTaskId: (id: string | null) => void
  reset: () => void
}

export const useDetectionStore = create<DetectionState>((set) => ({
  detections: [],
  inferenceTime: 0,
  selectedIndex: null,
  conf: 0.25,
  iou: 0.5,
  mode: 'idle',
  videoProgress: 0,
  videoTaskId: null,
  setDetections: (d, time) => set({ detections: d, inferenceTime: time }),
  setSelectedIndex: (i) => set({ selectedIndex: i }),
  setConf: (v) => set({ conf: v }),
  setIou: (v) => set({ iou: v }),
  setMode: (m) => set({ mode: m }),
  setVideoProgress: (p) => set({ videoProgress: p }),
  setVideoTaskId: (id) => set({ videoTaskId: id }),
  reset: () => set({ detections: [], inferenceTime: 0, selectedIndex: null, videoProgress: 0, videoTaskId: null, mode: 'idle' }),
}))
