interface SliderProps {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}

export default function Slider({ label, value, onChange, min = 0.01, max = 1.0, step = 0.01 }: SliderProps) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-600 font-medium w-10 flex-shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1"
      />
      <span className="text-xs text-emerald-600 w-12 text-right font-mono font-medium">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  )
}
