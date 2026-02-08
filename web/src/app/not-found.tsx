export default function NotFound() {
  return (
    <main className="min-h-screen bg-black text-zinc-100 flex items-center justify-center px-6">
      <div className="max-w-xl w-full text-center">
        <div className="inline-flex items-center justify-center px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] font-black uppercase tracking-[0.25em] text-zinc-400">
          404
        </div>
        <h1 className="mt-6 text-3xl md:text-4xl font-black tracking-tight text-white">
          Page not found
        </h1>
        <p className="mt-3 text-sm text-zinc-400 leading-relaxed">
          The page you&apos;re looking for doesn&apos;t exist or was moved.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <a
            href="/"
            className="px-5 py-3 rounded-2xl text-xs font-black uppercase tracking-wider bg-white text-black hover:bg-zinc-200 transition-colors"
          >
            Home
          </a>
          <a
            href="/admin"
            className="px-5 py-3 rounded-2xl text-xs font-black uppercase tracking-wider bg-white/5 text-zinc-200 border border-white/10 hover:bg-white/10 transition-colors"
          >
            Admin
          </a>
        </div>
      </div>
    </main>
  );
}
