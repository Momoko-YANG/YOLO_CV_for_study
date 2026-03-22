import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAuthStore } from '../../hooks/useAuth'
import { useNavigate } from 'react-router-dom'

interface SidebarProps {
  onCamera: () => void
  onImage: () => void
  onVideo: () => void
  onFolder: () => void
  models: string[]
  currentModel: string
  onModelSwitch: (model: string) => void
  mode: string
}

export default function Sidebar({ onCamera, onImage, onVideo, onFolder, models, currentModel, onModelSwitch, mode }: SidebarProps) {
  const [expanded, setExpanded] = useState(true)
  const [showPasswordModal, setShowPasswordModal] = useState(false)
  const [oldPassword, setOldPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordError, setPasswordError] = useState('')
  const [passwordSuccess, setPasswordSuccess] = useState('')
  const [isChangingPassword, setIsChangingPassword] = useState(false)
  const { t } = useTranslation()
  const logout = useAuthStore((s) => s.logout)
  const changePassword = useAuthStore((s) => s.changePassword)
  const user = useAuthStore((s) => s.user)
  const navigate = useNavigate()

  const buttons = [
    { icon: '📷', label: t('camera'), onClick: onCamera, active: mode === 'camera' },
    { icon: '🖼️', label: t('image'), onClick: onImage, active: mode === 'image' },
    { icon: '🎬', label: t('video'), onClick: onVideo, active: mode === 'video' },
    { icon: '📁', label: t('folder'), onClick: onFolder, active: mode === 'folder' },
  ]

  const closePasswordModal = () => {
    setShowPasswordModal(false)
    setOldPassword('')
    setNewPassword('')
    setConfirmPassword('')
    setPasswordError('')
  }

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!oldPassword || !newPassword || !confirmPassword) {
      setPasswordError(t('incomplete_info'))
      return
    }

    if (newPassword !== confirmPassword) {
      setPasswordError(t('password_mismatch'))
      return
    }

    setIsChangingPassword(true)
    setPasswordError('')

    try {
      await changePassword(oldPassword, newPassword)
      closePasswordModal()
      setPasswordSuccess(t('password_changed'))
      window.setTimeout(() => setPasswordSuccess(''), 3000)
    } catch (err: any) {
      setPasswordError(err.response?.data?.detail || t('request_failed'))
    } finally {
      setIsChangingPassword(false)
    }
  }

  return (
    <>
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

          {/* Model selector */}
          <div className="flex items-center gap-3 px-3 py-3 rounded-lg text-sm text-gray-600">
            <span className="text-lg flex-shrink-0">🧠</span>
            {expanded && (
              <select
                value={currentModel}
                onChange={(e) => onModelSwitch(e.target.value)}
                className="flex-1 min-w-0 bg-white border border-gray-200 rounded-md px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-emerald-400 cursor-pointer"
              >
                {models.map((m) => (
                  <option key={m} value={m}>{m.replace('.pt', '')}</option>
                ))}
              </select>
            )}
          </div>

          {expanded && passwordSuccess && (
            <div className="mx-2 mt-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              {passwordSuccess}
            </div>
          )}
        </div>

        {/* User section */}
        <div className="border-t border-emerald-100 p-2 space-y-1">
          <button
            type="button"
            className="flex items-center gap-3 px-3 py-3 rounded-lg text-sm text-gray-600 w-full"
          >
            <span className="text-lg">👤</span>
            {expanded && <span className="whitespace-nowrap truncate">{user?.username}</span>}
          </button>

          {expanded && (
            <button
              onClick={() => {
                setPasswordSuccess('')
                setPasswordError('')
                setShowPasswordModal(true)
              }}
              className="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-gray-100 transition-colors text-sm text-gray-600 w-full"
            >
              <span className="text-lg">🔒</span>
              <span className="whitespace-nowrap">{t('change_password')}</span>
            </button>
          )}

          <button
            onClick={() => { logout(); navigate('/login') }}
            className="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-gray-100 transition-colors text-sm text-gray-600 w-full"
          >
            <span className="text-lg">↩️</span>
            {expanded && <span className="whitespace-nowrap">{t('logout')}</span>}
          </button>
        </div>
      </div>

      {showPasswordModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
          <div className="w-full max-w-sm rounded-2xl border border-emerald-100 bg-white p-6 shadow-xl shadow-emerald-100/50">
            <div className="mb-4">
              <h2 className="text-lg font-semibold text-gray-800">{t('change_password')}</h2>
              <p className="mt-1 text-sm text-gray-500">{user?.username}</p>
            </div>

            <form onSubmit={handlePasswordChange} className="space-y-3">
              <input
                type="password"
                value={oldPassword}
                onChange={(e) => setOldPassword(e.target.value)}
                placeholder={t('old_password')}
                maxLength={32}
                className="w-full rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-700 outline-none transition-all focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder={t('new_password')}
                maxLength={32}
                className="w-full rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-700 outline-none transition-all focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder={t('confirm_password')}
                maxLength={32}
                className="w-full rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-700 outline-none transition-all focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />

              {passwordError && (
                <div className="rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600">
                  {passwordError}
                </div>
              )}

              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={closePasswordModal}
                  className="flex-1 rounded-lg border border-gray-200 px-4 py-3 text-sm text-gray-600 transition-colors hover:bg-gray-50"
                >
                  {t('cancel')}
                </button>
                <button
                  type="submit"
                  disabled={isChangingPassword}
                  className="flex-1 rounded-lg bg-emerald-500 px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-emerald-600 disabled:opacity-50"
                >
                  {isChangingPassword ? t('saving') : t('change_password')}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  )
}
