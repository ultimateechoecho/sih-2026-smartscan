export default function Panel({ title, right, children, className = '', bodyClass = '' }) {
  return (
    <section className={`panel flex flex-col ${className}`}>
      {title && (
        <header className="flex items-center justify-between px-3 pt-2.5 pb-1">
          <h2 className="panel-title">{title}</h2>
          {right}
        </header>
      )}
      <div className={`flex-1 min-h-0 px-3 pb-3 ${bodyClass}`}>{children}</div>
    </section>
  )
}
