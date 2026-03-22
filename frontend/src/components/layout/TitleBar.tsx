import { useTranslation } from 'react-i18next'

interface TitleBarProps {
  onSave: () => void
}

export default function TitleBar({ onSave }: TitleBarProps) {
  const { t, i18n } = useTranslation()

  const toggleLang = () => {
    const next = i18n.language === 'en' ? 'ja' : 'en'
    i18n.changeLanguage(next)
    localStorage.setItem('language', next)
  }

  return (
    <div className="h-12 bg-white/80 backdrop-blur-sm border-b border-emerald-100 flex items-center px-4 gap-3 shadow-sm">
      <h1 className="flex-1 text-center text-sm font-semibold text-emerald-700 tracking-wide">
        {t('app_title')}
      </h1>
      <button
        onClick={toggleLang}
        className="px-3 py-1 text-xs rounded-md bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors font-medium"
      >
        {i18n.language === 'en' ? '日本語' : 'EN'}
      </button>
      <button
        onClick={onSave}
        className="px-3 py-1 text-xs rounded-md bg-emerald-500 text-white hover:bg-emerald-600 transition-colors font-medium shadow-sm"
      >
        {t('save')}
      </button>
    </div>
  )
}
