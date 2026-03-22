import { useTranslation } from 'react-i18next'

interface TitleBarProps {
  onSave: () => void
  canExport: boolean
}

export default function TitleBar({ onSave, canExport }: TitleBarProps) {
  const { t, i18n } = useTranslation()

  const handleLanguageChange = (next: string) => {
    i18n.changeLanguage(next)
    localStorage.setItem('language', next)
  }

  return (
    <div className="h-12 bg-white/80 backdrop-blur-sm border-b border-emerald-100 flex items-center px-4 gap-3 shadow-sm">
      <h1 className="flex-1 text-center text-sm font-semibold text-emerald-700 tracking-wide">
        {t('app_title')}
      </h1>
      <select
        value={i18n.resolvedLanguage || i18n.language}
        onChange={(e) => handleLanguageChange(e.target.value)}
        className="rounded-md bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600 outline-none transition-colors hover:bg-gray-200"
      >
        <option value="zh">中文</option>
        <option value="en">EN</option>
        <option value="ja">日本語</option>
      </select>
      <button
        onClick={onSave}
        disabled={!canExport}
        className="px-3 py-1 text-xs rounded-md bg-emerald-500 text-white hover:bg-emerald-600 transition-colors font-medium shadow-sm disabled:cursor-not-allowed disabled:bg-emerald-200 disabled:text-white/80"
      >
        {t('export_csv')}
      </button>
    </div>
  )
}
