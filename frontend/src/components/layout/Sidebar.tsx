import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAuthStore } from '../../hooks/useAuth'
import { useNavigate } from 'react-router-dom'

interface SidebarProps {
  onCamera: () => void
  onImage: () => void
  onVideo: () => void
  onFolder: () => void
  onModel: () => void
  mode: string
}

export default function Sidebar({ onCamera, onImage, onVideo, onFolder, onModel, mode }: SidebarProps) {
  const [expanded, setExpanded] = useState(true)
  const { t } = useTranslation()
  const logout = useAuthStore((s) => s.logout)
  const user = useAuthStore((s) => s.user)
  const navigate = useNavigate()

  const buttons = [
    { icon: '📷', label: t('camera'), onClick: onCamera, active: mode === 'camera' },
    { icon: '🖼️', label: t('image'), onClick: onImage, active: mode === 'image' },
    { icon: '🎬', label: t('video'), onClick: onVideo, active: mode === 'video' },
    { icon: '📁', label: t('folder'), onClick: onFolder, active: mode === 'folder' },
    { icon: '🧠', label: t('model'), onClick: onModel, active: false },
  ]

  return (
    <div
      className="h-full flex flex-col bg-white/80 backdrop-blur-sm border-r border-emerald-100 transition-all duration-[400ms] overflow-hidden shadow-sm"
      style={{ width: expanded ? 240 : 55 }}
    >
      {/* Menu toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full h-12 flex items-center justify-center text-xl text-gray-500 hover:bg-emerald-50 hover:text-emerald-600 transition-colors"
      >
        {expanded ? '✕' : '☰'}
      </button>

      {/* Action buttons */}
      <div className="flex-1 flex flex-col gap-1 px-1.5 mt-2">
        {buttons.map((btn) => (
          <button
            key={btn.label}
            onClick={btn.onClick}
            className={`flex items-center gap-3 px-3 py-3 rounded-lg transition-colors text-sm
              ${btn.active ? 'bg-emerald-100 text-emerald-700 font-medium' : 'hover:bg-gray-100 text-gray-600'}`}
          >
            <span className="text-lg flex-shrink-0">{btn.icon}</span>
            {expanded && <span className="whitespace-nowrap">{btn.label}</span>}
          </button>
        ))}
      </div>

      {/* User section */}
      <div className="border-t border-emerald-100 p-2">
        <button
          onClick={() => { logout(); navigate('/login') }}
          className="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-gray-100 transition-colors text-sm text-gray-600 w-full"
        >
          <span className="text-lg">👤</span>
          {expanded && <span className="whitespace-nowrap truncate">{user?.username || t('logout')}</span>}
        </button>
      </div>
    </div>
  )
}
