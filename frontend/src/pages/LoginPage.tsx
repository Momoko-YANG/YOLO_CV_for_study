import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuthStore } from '../hooks/useAuth'

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { t } = useTranslation()
  const login = useAuthStore((s) => s.login)
  const register = useAuthStore((s) => s.register)
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username || !password) {
      setError(t('incomplete_info'))
      return
    }
    setError('')
    setLoading(true)
    try {
      if (isLogin) {
        await login(username, password)
      } else {
        await register(username, password)
      }
      navigate('/')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-emerald-50 via-white to-green-50">
      <div className="w-full max-w-md bg-white rounded-2xl border border-emerald-100 p-8 shadow-lg shadow-emerald-100/50">
        {/* Title */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto mb-4 bg-emerald-50 rounded-2xl flex items-center justify-center text-4xl">✋</div>
          <h1 className="text-xl font-semibold text-emerald-700">{t('app_title')}</h1>
        </div>

        {/* Tab switch */}
        <div className="flex mb-6 bg-emerald-50 rounded-lg p-1">
          <button
            onClick={() => { setIsLogin(true); setError('') }}
            className={`flex-1 py-2 text-sm rounded-md transition-colors
              ${isLogin ? 'bg-emerald-500 text-white font-medium shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
          >
            {t('login')}
          </button>
          <button
            onClick={() => { setIsLogin(false); setError('') }}
            className={`flex-1 py-2 text-sm rounded-md transition-colors
              ${!isLogin ? 'bg-emerald-500 text-white font-medium shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
          >
            {t('register')}
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder={t('username')}
              maxLength={10}
              className="w-full bg-gray-50 border border-gray-200 rounded-lg px-4 py-3 text-sm text-gray-700 placeholder-gray-400 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all"
            />
          </div>
          <div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={t('password')}
              maxLength={12}
              className="w-full bg-gray-50 border border-gray-200 rounded-lg px-4 py-3 text-sm text-gray-700 placeholder-gray-400 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all"
            />
          </div>

          {error && (
            <div className="text-red-500 text-xs text-center bg-red-50 py-2 rounded-lg">{error}</div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-emerald-500 text-white font-medium py-3 rounded-lg hover:bg-emerald-600 transition-colors disabled:opacity-50 shadow-sm shadow-emerald-200"
          >
            {loading ? t('logging_in') : isLogin ? t('login') : t('register')}
          </button>
        </form>

        <div className="text-center mt-4">
          <button
            onClick={() => { setIsLogin(!isLogin); setError('') }}
            className="text-xs text-gray-400 hover:text-emerald-600 transition-colors"
          >
            {isLogin ? t('go_to_register') : t('go_to_login')}
          </button>
        </div>
      </div>
    </div>
  )
}
