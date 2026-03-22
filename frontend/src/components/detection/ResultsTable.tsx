import { useTranslation } from 'react-i18next'
import type { DetectionResult } from '../../types/detection'

interface ResultsTableProps {
  detections: DetectionResult[]
  selectedIndex: number | null
  onSelect: (index: number) => void
}

export default function ResultsTable({ detections, selectedIndex, onSelect }: ResultsTableProps) {
  const { t } = useTranslation()

  return (
    <div className="bg-white rounded-lg border border-emerald-100 overflow-hidden shadow-sm">
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-emerald-50 text-emerald-700">
            <th className="py-2 px-2 text-left w-10 font-semibold">ID</th>
            <th className="py-2 px-3 text-left font-semibold">{t('class')}</th>
            <th className="py-2 px-3 text-left font-semibold">{t('coordinates')}</th>
            <th className="py-2 px-3 text-right font-semibold">{t('confidence')}</th>
          </tr>
        </thead>
        <tbody>
          {detections.length === 0 && (
            <tr>
              <td colSpan={4} className="py-4 text-center text-gray-400 text-xs">
                {t('no_detections')}
              </td>
            </tr>
          )}
          {detections.map((det, i) => (
            <tr
              key={i}
              onClick={() => onSelect(i)}
              className={`cursor-pointer transition-colors border-t border-gray-100
                ${selectedIndex === i ? 'bg-emerald-50' : 'hover:bg-gray-50'}`}
            >
              <td className="py-1.5 px-2 text-gray-400">{i + 1}</td>
              <td className="py-1.5 px-3 text-gray-700 font-medium">{det.class_name}</td>
              <td className="py-1.5 px-3 text-gray-400 font-mono">
                [{det.bbox.join(', ')}]
              </td>
              <td className="py-1.5 px-3 text-right text-emerald-600 font-medium">
                {(det.score * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
