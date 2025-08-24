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
    <div className="flex min-h-screen">
      {/* Left side - Travel landscape */}
      <div
        className="hidden lg:flex lg:flex-1 relative bg-cover bg-center"
        style={{
          backgroundImage: `url('https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-nZOfqmUBmBE42qobfmZfxAIIzKmnNB.png')`,
        }}
      >
        <div className="absolute inset-0 bg-black/30" />
        <div className="relative z-10 flex flex-col justify-end p-12 text-white">
          <h1 className="text-5xl font-bold mb-4 leading-tight">
            Collect memories,
            <br />
            not things.
          </h1>
          <p className="text-lg opacity-90 max-w-md mb-4">
            Travel isn't just about seeing new places, it's about gaining new perspectives
          </p>
          <p className="text-base opacity-80 max-w-md">Explore not only the world outside, but also the world within</p>
        </div>
      </div>

      {/* Right side - Auth form */}
      <div className="flex-1 flex items-center justify-center bg-white px-4 py-12 sm:px-6 lg:px-8">
        <div className="w-full max-w-md space-y-4">
          {searchParams?.error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              {decodeURIComponent(searchParams.error)}
            </div>
          )}
          {searchParams?.success && (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg text-sm">
              {decodeURIComponent(searchParams.success)}
            </div>
          )}
          <AuthForm />
        </div>
      </div>
    </div>
  )
}
