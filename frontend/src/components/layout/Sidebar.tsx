import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../../hooks/useAuth'
import api from '../../services/api'

interface SidebarProps {
  onCamera: () => void
  onImage: () => void
  onVideo: () => void
  onFolder: () => void
  onModel: () => void
  mode: string
}

function getErrorDetail(err: unknown): string | undefined {
  if (typeof err !== 'object' || err === null || !('response' in err)) return undefined
  const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail
  return typeof detail === 'string' ? detail : undefined
}

export default function Sidebar({ onCamera, onImage, onVideo, onFolder, onModel, mode }: SidebarProps) {
  const [expanded, setExpanded] = useState(true)
  const [showChangePassword, setShowChangePassword] = useState(false)
  const [oldPassword, setOldPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)
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

  const closeDialog = () => {
    setShowChangePassword(false)
    setOldPassword('')
    setNewPassword('')
    setConfirmPassword('')
    setError('')
    setSuccess('')
    setLoading(false)
  }

  const handleChangePassword = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!oldPassword || !newPassword || !confirmPassword) {
      setError(t('incomplete_info'))
      setSuccess('')
      return
    }

    if (newPassword !== confirmPassword) {
      setError(t('password_mismatch'))
      setSuccess('')
      return
    }

    setLoading(true)
    setError('')
    setSuccess('')

    try {
      await api.put('/api/auth/change-password', {
        old_password: oldPassword,
        new_password: newPassword,
      })
      setSuccess(t('password_changed'))
      setOldPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err: unknown) {
      setError(getErrorDetail(err) || t('save_failed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div
        className="h-full flex flex-col bg-white/80 backdrop-blur-sm border-r border-emerald-100 transition-all duration-[400ms] overflow-hidden shadow-sm"
        style={{ width: expanded ? 240 : 55 }}
      >
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full h-12 flex items-center justify-center text-xl text-gray-500 hover:bg-emerald-50 hover:text-emerald-600 transition-colors"
        >
          {expanded ? '✕' : '☰'}
        </button>

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

        <div className="border-t border-emerald-100 p-2">
          <button
            onClick={() => setShowChangePassword(true)}
            className="mb-1 flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-amber-50 transition-colors text-sm text-amber-700 w-full"
          >
            <span className="text-lg">🔐</span>
            {expanded && <span className="whitespace-nowrap">{t('change_password')}</span>}
          </button>
          <button
            onClick={() => { logout(); navigate('/login') }}
            className="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-gray-100 transition-colors text-sm text-gray-600 w-full"
          >
            <span className="text-lg">👤</span>
            {expanded && <span className="whitespace-nowrap truncate">{user?.username || t('logout')}</span>}
          </button>
        </div>
      </div>

      {showChangePassword && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/35 p-4">
          <div className="w-full max-w-md rounded-2xl border border-emerald-100 bg-white p-6 shadow-xl">
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800">{t('change_password')}</h2>
              <button
                onClick={closeDialog}
                className="rounded-md px-2 py-1 text-sm text-slate-400 hover:bg-slate-100 hover:text-slate-600"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleChangePassword} className="space-y-3">
              <input
                type="password"
                value={oldPassword}
                onChange={(e) => setOldPassword(e.target.value)}
                placeholder={t('current_password')}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition-all focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder={t('new_password')}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition-all focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder={t('confirm_password')}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition-all focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />

              {error && (
                <div className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600">
                  {error}
                </div>
              )}

              {success && (
                <div className="rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
                  {success}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={closeDialog}
                  className="rounded-lg border border-slate-200 px-4 py-2 text-sm text-slate-600 hover:bg-slate-50"
                >
                  {t('cancel')}
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
                >
                  {loading ? t('processing') : t('update_password')}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  )
}
