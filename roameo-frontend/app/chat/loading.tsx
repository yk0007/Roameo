"use client"

export default function Loading() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-white relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-slate-50/50 rounded-full blur-3xl -z-10"></div>
      
      <div className="text-center w-full max-w-sm mx-auto p-12 bg-white/60 backdrop-blur-2xl rounded-[32px] shadow-[0_20px_80px_rgba(15,23,42,0.06),_0_6px_20px_rgba(15,23,42,0.04)] border border-slate-100/60 transition-all">
        {/* Roameo Logo Animation */}
        <div className="mb-10 relative z-10">
          <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto relative overflow-hidden shadow-[0_10px_30px_rgba(15,23,42,0.15)] ring-4 ring-white">
            <div className="w-6 h-6 bg-white rounded-full animate-pulse shadow-[0_0_20px_rgba(255,255,255,0.8)]"></div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20 animate-sweep"></div>
          </div>
          <div className="absolute inset-0 w-20 h-20 bg-slate-900 rounded-full mx-auto opacity-10 animate-ping"></div>
        </div>
        
        {/* Roameo Text */}
        <h1 className="text-3xl font-bold text-slate-900 mb-2 animate-fade-in tracking-tight">
          Roameo
        </h1>
        <p className="text-slate-400 mb-8 animate-fade-in-delay text-[15px] font-medium">
          Preparing your workspace...
        </p>
        
        {/* Enhanced Loading Animation */}
        <div className="mb-8">
          <div className="flex justify-center items-center space-x-2.5 mb-6">
            <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-slate-900 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
          
          {/* Progress indicator */}
          <div className="w-32 h-[3px] bg-slate-100 rounded-full mx-auto overflow-hidden">
            <div className="w-full h-full bg-slate-800 rounded-full animate-progress"></div>
          </div>
        </div>
        
        <div className="space-y-3 text-xs text-slate-400 font-medium">
          <p className="animate-pulse flex items-center justify-center gap-2.5"><span className="text-[16px] text-slate-700">🧠</span> Initializing agents</p>
          <p className="animate-pulse flex items-center justify-center gap-2.5" style={{animationDelay: '0.5s'}}><span className="text-[16px] text-slate-700">🗺️</span> Loading maps</p>
          <p className="animate-pulse flex items-center justify-center gap-2.5" style={{animationDelay: '1s'}}><span className="text-[16px] text-slate-700">✨</span> Organizing interface</p>
        </div>
      </div>
      
      <style jsx>{`
        @keyframes sweep {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-sweep {
          animation: sweep 2s ease-in-out infinite;
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }
        .animate-fade-in-delay {
          animation: fade-in 1s ease-out 0.4s both;
        }
        @keyframes progress {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-progress {
          animation: progress 1.5s ease-in-out infinite;
        }
      `}</style>
    </div>
  )
}
