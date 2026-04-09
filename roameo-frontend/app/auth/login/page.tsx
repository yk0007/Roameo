import type { Metadata } from "next"
import { createClient, isSupabaseConfigured } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import AuthForm from "@/components/auth-form"
import ClearQuery from "@/components/clear-query"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import { AuthVisualPanel } from "@/components/blocks/auth-visual-panel"
import { EntranceMotion, SectionReveal } from "@/components/ui/site-motion"

export const metadata: Metadata = {
  title: "Log In",
}

export default async function LoginPage({
  searchParams,
}: {
  searchParams?: Promise<{ error?: string; success?: string }>
}) {
  const resolvedSearchParams = await searchParams

  // If Supabase is not configured, show setup message directly
  if (!isSupabaseConfigured) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50">
        <h1 className="text-2xl font-bold mb-4 text-gray-900">Connect Supabase to get started</h1>
      </div>
    )
  }

  // Check if user is already logged in
  const supabase = await createClient()
  const {
    data: { session },
  } = await supabase.auth.getSession()

  // If user is logged in, redirect to dashboard
  if (session) {
    redirect("/dashboard")
  }

  return (
    <div className="h-screen overflow-hidden bg-[linear-gradient(180deg,#f9fcff_0%,#f4f8ff_100%)]">
      {/* Clear query params after first render so banners show only once */}
      {/* This is a client-only helper and renders nothing visually */}
      <ClearQuery keys={["success"]} />
      <div className="relative h-screen overflow-hidden px-0 py-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(186,230,253,0.45),transparent_24%),radial-gradient(circle_at_bottom_left,rgba(219,234,254,0.72),transparent_30%)]" />
        <div className="relative z-10 flex h-screen">
          <SectionReveal className="hidden h-screen w-[53%] lg:block" delay={0.04}>
            <AuthVisualPanel />
          </SectionReveal>

          <div className="relative flex w-full flex-col overflow-hidden bg-transparent px-6 py-6 lg:w-[47%] lg:px-8 lg:py-7 xl:px-10">
            <img
              src="https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&fm=jpg&q=80&w=1800"
              alt=""
              aria-hidden="true"
              className="absolute inset-0 h-full w-full object-cover scale-110 blur-[10px]"
            />
            <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(248,251,255,0.72)_0%,rgba(240,246,255,0.76)_100%)] backdrop-blur-xl" />

            <EntranceMotion className="flex items-center justify-start" delay={0.06}>
              <Link
                href="/"
                className="relative z-10 inline-flex items-center gap-2 rounded-full border border-white/45 bg-white/55 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:border-white/60 hover:text-slate-950"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to home
              </Link>
            </EntranceMotion>

            <div className="relative z-10 flex flex-1 items-center justify-center">
              <EntranceMotion
                className="w-full max-w-[29rem] rounded-[32px] border border-white/45 bg-white/32 p-8 shadow-[0_28px_80px_rgba(15,23,42,0.12)] backdrop-blur-2xl"
                delay={0.12}
              >
                {resolvedSearchParams?.error && (
                  <div className="mb-5 border border-rose-200 bg-rose-50/90 px-4 py-3 text-sm text-rose-700">
                    {decodeURIComponent(resolvedSearchParams.error)}
                  </div>
                )}
                {resolvedSearchParams?.success && (
                  <div className="mb-5 border border-emerald-200 bg-emerald-50/90 px-4 py-3 text-sm text-emerald-700">
                    {decodeURIComponent(resolvedSearchParams.success)}
                  </div>
                )}

                <AuthForm />
              </EntranceMotion>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
