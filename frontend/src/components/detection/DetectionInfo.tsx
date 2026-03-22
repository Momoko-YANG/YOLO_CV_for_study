import { useTranslation } from 'react-i18next'
import type { DetectionResult } from '../../types/detection'

interface DetectionInfoProps {
  detections: DetectionResult[]
  inferenceTime: number
  selectedIndex: number | null
  onSelectTarget: (index: number | null) => void
}

export default function DetectionInfo({ detections, inferenceTime, selectedIndex, onSelectTarget }: DetectionInfoProps) {
  const { t } = useTranslation()
  const selected = selectedIndex !== null ? detections[selectedIndex] : null

  return (
    <div className="space-y-3">
      {/* Stats row */}
      <div className="flex gap-3">
        <div className="flex-1 bg-white rounded-lg p-3 border border-emerald-100 shadow-sm">
          <div className="text-[10px] text-gray-400 uppercase tracking-wider">{t('inference_time')}</div>
          <div className="text-lg font-mono text-emerald-600 font-semibold">
            {inferenceTime > 0 ? `${(inferenceTime * 1000).toFixed(0)}ms` : '--'}
          </div>
        </div>
        <div className="flex-1 bg-white rounded-lg p-3 border border-emerald-100 shadow-sm">
          <div className="text-[10px] text-gray-400 uppercase tracking-wider">{t('total_count')}</div>
          <div className="text-lg font-mono text-emerald-600 font-semibold">{detections.length}</div>
        </div>
      </div>

      {/* Target selector */}
      <select
        value={selectedIndex ?? ''}
        onChange={(e) => onSelectTarget(e.target.value === '' ? null : Number(e.target.value))}
        className="w-full bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all"
      >
        <option value="">{t('all_targets')}</option>
        {detections.map((det, i) => (
          <option key={i} value={i}>
            {det.class_name} - {(det.score * 100).toFixed(1)}%
          </option>
        ))}
      </select>

      {/* Selected detection details */}
      {selected && (
        <div className="bg-white rounded-lg p-3 border border-emerald-100 space-y-2 text-xs shadow-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">{t('class')}</span>
            <span className="text-gray-700 font-medium">{selected.class_name}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">{t('confidence')}</span>
            <span className="text-emerald-600 font-medium">{(selected.score * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">{t('coordinates')}</span>
            <span className="text-gray-500 font-mono">[{selected.bbox.join(', ')}]</span>
          </div>
        </div>
      )}
    </div>
  )
}
