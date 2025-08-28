import { createClient, isSupabaseConfigured } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import AuthForm from "@/components/auth-form"

export default async function LoginPage({
  searchParams,
}: {
  searchParams?: { error?: string; success?: string }
}) {
  // If Supabase is not configured, show setup message directly
  if (!isSupabaseConfigured) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50">
        <h1 className="text-2xl font-bold mb-4 text-gray-900">Connect Supabase to get started</h1>
      </div>
    )
  }

  // Check if user is already logged in
  const supabase = createClient()
  const {
    data: { session },
  } = await supabase.auth.getSession()

  // If user is logged in, redirect to dashboard
  if (session) {
    redirect("/dashboard")
  }

  return (
    <div className="min-h-screen bg-white flex">
      {/* Left side - Auth Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
          {/* Error/Success Messages */}
          {searchParams?.error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm mb-6">
              {decodeURIComponent(searchParams.error)}
            </div>
          )}
          {searchParams?.success && (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg text-sm mb-6">
              {decodeURIComponent(searchParams.success)}
            </div>
          )}
          
          <AuthForm />
        </div>
      </div>

      {/* Right side - Travel Illustration */}
      <div className="hidden lg:flex flex-1 bg-gradient-to-br from-teal-400 via-cyan-400 to-blue-500 items-center justify-center relative overflow-hidden">
        {/* Travel Illustration - Full Screen */}
        <div className="absolute inset-0 w-full h-full">
          <img
            src="/travel-illustration.png"
            alt="Travel illustration with inspirational quotes about collecting memories and exploring the world"
            className="w-full h-full object-cover"
          />
        </div>
      </div>
    </div>
  )
}
