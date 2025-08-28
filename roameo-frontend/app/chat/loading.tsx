"use client"

export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="text-center max-w-md mx-auto px-6">
        {/* Roameo Logo Animation */}
        <div className="mb-8 relative">
          <div className="w-24 h-24 bg-black rounded-full flex items-center justify-center mx-auto mb-4 relative overflow-hidden">
            <div className="w-10 h-10 bg-white rounded-full animate-pulse"></div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-sweep"></div>
          </div>
          <div className="absolute inset-0 w-24 h-24 bg-black rounded-full mx-auto opacity-10 animate-ping"></div>
          <div className="absolute inset-0 w-24 h-24 bg-blue-500 rounded-full mx-auto opacity-5 animate-ping" style={{animationDelay: '0.5s'}}></div>
        </div>
        
        {/* Roameo Text */}
        <h1 className="text-4xl font-bold text-gray-900 mb-3 animate-fade-in">
          roameo
        </h1>
        <p className="text-gray-600 mb-8 animate-fade-in-delay text-lg">
          Preparing your AI travel companion...
        </p>
        
        {/* Enhanced Loading Animation */}
        <div className="mb-6">
          <div className="flex justify-center items-center space-x-3 mb-4">
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-3 h-3 bg-blue-700 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
          
          {/* Progress indicator */}
          <div className="w-48 h-1 bg-gray-200 rounded-full mx-auto overflow-hidden">
            <div className="w-full h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full animate-progress"></div>
          </div>
        </div>
        
        <div className="space-y-2 text-sm text-gray-500">
          <p className="animate-pulse">🧠 Initializing AI agents...</p>
          <p className="animate-pulse" style={{animationDelay: '0.5s'}}>🗺️ Loading travel database...</p>
          <p className="animate-pulse" style={{animationDelay: '1s'}}>✨ Personalizing your experience...</p>
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
          from { opacity: 0; transform: translateY(15px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 1.2s ease-out;
        }
        .animate-fade-in-delay {
          animation: fade-in 1.2s ease-out 0.6s both;
        }
        @keyframes progress {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(0%); }
        }
        .animate-progress {
          animation: progress 2s ease-in-out infinite;
        }
      `}</style>
    </div>
  )
}
