interface ProgressBarProps {
  progress: number // 0-100
  visible: boolean
}

export default function ProgressBar({ progress, visible }: ProgressBarProps) {
  if (!visible) return null

  return (
    <div className="w-full bg-emerald-100 rounded-full h-2 overflow-hidden">
      <div
        className="h-full bg-emerald-500 transition-all duration-300 rounded-full"
        style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
      />
    </div>
  )
}
