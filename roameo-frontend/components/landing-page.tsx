"use client"

import { Button } from "@/components/ui/button"
import {
  ArrowRight,
  BellRing,
  CalendarClock,
  Compass,
  Gem,
  Landmark,
  MapPin,
  Sparkles,
  WalletCards,
} from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import { supabase } from "@/lib/supabase/client"
import type { AuthChangeEvent, Session, User } from "@supabase/supabase-js"
import { FeaturesSectionWithHoverEffects } from "@/components/blocks/feature-section-with-hover-effects"
import { LandingTripMoods } from "@/components/blocks/landing-trip-moods"
import { RotatingFeatureDial } from "@/components/blocks/rotating-feature-dial"
import HeroScrollAnimation from "@/components/ui/hero-scroll-animation"

const featureHighlights = [
  {
    title: "Local Expertise",
    description:
      "Navigate India's diverse regions with insider knowledge. Get authentic recommendations from local experts across hill stations and coastal areas.",
    icon: Compass,
    wrapperClass:
      "rounded-full bg-gradient-to-br from-blue-400 to-purple-600",
  },
  {
    title: "Hidden Gems",
    description:
      "Discover secret viewpoints in Ooty, untouched waterfalls in Araku, and local festivals that typical guides miss.",
    icon: Gem,
    wrapperClass:
      "rounded-2xl bg-gradient-to-br from-emerald-400 to-green-600",
  },
  {
    title: "Smart Budgeting",
    description:
      "From budget stays in Darjeeling to luxury resorts in Munnar, get cost-effective recommendations that maximize value.",
    icon: WalletCards,
    wrapperClass:
      "rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-600",
  },
  {
    title: "Cultural Sensitivity",
    description:
      "Navigate India's rich cultural landscape with confidence. Get guidance on local customs and regional traditions.",
    icon: Landmark,
    wrapperClass:
      "rounded-2xl bg-gradient-to-br from-rose-400 to-pink-600",
  },
  {
    title: "Seasonal Planning",
    description:
      "Plan around monsoons and peak seasons. Visit Munnar during tea harvest or RK Beach during pleasant winters.",
    icon: CalendarClock,
    wrapperClass:
      "rounded-2xl bg-gradient-to-br from-amber-400 to-orange-600",
  },
  {
    title: "Real-time Updates",
    description:
      "Stay informed about weather conditions in hill stations, local events, and travel advisories for smooth journeys.",
    icon: BellRing,
    wrapperClass:
      "rounded-full bg-gradient-to-br from-green-400 via-emerald-500 to-teal-700",
  },
] as const

export function LandingPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [navOnDark, setNavOnDark] = useState(true)

  useEffect(() => {
    // Check current user session
    const checkUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession()
      setUser(session?.user || null)
    }

    checkUser()

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session: Session | null) => {
      setUser(session?.user || null)
    })

    return () => subscription.unsubscribe()
  }, [])

  useEffect(() => {
    const updateNavTone = () => {
      const heroBoundary = window.innerHeight * 1.1
      setNavOnDark(window.scrollY < heroBoundary)
    }

    updateNavTone()
    window.addEventListener("scroll", updateNavTone, { passive: true })
    window.addEventListener("resize", updateNavTone)

    return () => {
      window.removeEventListener("scroll", updateNavTone)
      window.removeEventListener("resize", updateNavTone)
    }
  }, [])

  const handleProtectedAction = (action: string) => {
    if (!user) {
      router.push("/auth/login")
      return
    }
    router.push("/dashboard")
  }

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut()
    } catch (e) {
      // ignore errors and continue navigation
    } finally {
      setUser(null)
      // Hard navigation prevents intermediate renders/flicker
      window.location.replace("/auth/login")
    }
  }

  const navLinks = [
    { label: "How it works", href: "#how-it-works" },
    { label: "Features", href: "#features" },
  ] as const

  return (
    <div className="min-h-screen bg-white">
      {/* Mobile Notice - Only visible on small screens */}
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white p-6 md:hidden">
        <div className="text-center max-w-sm">
          <div className="w-16 h-16 bg-black rounded-full flex items-center justify-center mx-auto mb-6">
            <div className="w-4 h-4 bg-white rounded-full"></div>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-4">roameo</h1>
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Desktop Experience Required</h2>
          <p className="text-gray-600 mb-6 leading-relaxed">
            For the best experience, please visit our website on a desktop or laptop computer.
          </p>
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-4">
            <p className="text-orange-800 text-sm font-medium">
              📱 Mobile version coming soon!
            </p>
          </div>
          <p className="text-sm text-gray-500">
            We're working on an amazing mobile experience for you.
          </p>
        </div>
      </div>

      {/* Main content - Hidden on mobile */}
      <div className="hidden md:block">
        <header className="fixed left-1/2 top-5 z-50 w-full max-w-7xl -translate-x-1/2 px-4">
          <div
            className={`flex items-center justify-between rounded-[28px] px-6 py-4 backdrop-blur-xl backdrop-saturate-150 transition-all duration-300 ${
              navOnDark
                ? "bg-[#09143a]/24 shadow-[0_24px_70px_rgba(8,18,58,0.22)]"
                : "bg-white/72 shadow-[0_24px_60px_rgba(15,23,42,0.08)]"
            }`}
          >
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-black shadow-[0_10px_24px_rgba(0,0,0,0.18)]">
                <div className="h-2.5 w-2.5 rounded-full bg-white" />
              </div>
              <span className={`text-[1.75rem] font-semibold tracking-[-0.045em] transition-colors duration-300 ${navOnDark ? "text-white" : "text-slate-950"}`}>
                roameo
              </span>
            </div>

            <nav
              className={`hidden items-center rounded-full p-1 transition-all duration-300 md:flex ${
                navOnDark
                  ? "bg-white/8 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
                  : "bg-white/55 shadow-[inset_0_0_0_1px_rgba(226,232,240,0.9)]"
              }`}
            >
              {navLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  className={`rounded-full px-5 py-2 text-sm font-medium transition-colors ${
                    navOnDark
                      ? "text-white/78 hover:bg-white/10 hover:text-white"
                      : "text-slate-600 hover:bg-white hover:text-slate-950"
                  }`}
                >
                  {link.label}
                </a>
              ))}
            </nav>

            <div className="flex items-center gap-3">
              {user ? (
                <Button
                  onClick={handleSignOut}
                  variant="ghost"
                  className={`rounded-full px-5 transition-colors ${
                    navOnDark ? "text-white/78 hover:text-white hover:bg-white/10" : "text-slate-600 hover:text-slate-950"
                  }`}
                >
                  Sign out
                </Button>
              ) : (
                <Link href="/auth/login">
                  <Button
                    variant="ghost"
                    className={`rounded-full px-5 transition-colors ${
                      navOnDark ? "text-white/78 hover:text-white hover:bg-white/10" : "text-slate-600 hover:text-slate-950"
                    }`}
                  >
                    Log in
                  </Button>
                </Link>
              )}
              <Button
                onClick={() => handleProtectedAction("get started")}
                className="rounded-full bg-black px-6 text-white shadow-[0_14px_28px_rgba(0,0,0,0.16)] hover:bg-gray-800"
              >
                Get started
              </Button>
            </div>
          </div>
        </header>

      {/* Hero Section with Scroll Animation */}
      <HeroScrollAnimation user={user} handleProtectedAction={handleProtectedAction} />

      {/* How it Works Section */}
      <section id="how-it-works" className="px-6 min-h-screen flex items-center bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-900 mb-4 font-sans">How it Works</h2>
            <p className="text-xl text-gray-600">Discover the power of AI-driven travel planning</p>
          </div>
          <FeaturesSectionWithHoverEffects />
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative min-h-screen bg-white px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="max-w-2xl">
            <h2 className="text-5xl font-bold tracking-[-0.05em] text-gray-900 font-sans">Features</h2>
            <p className="mt-4 text-xl leading-8 text-gray-600 font-sans">
              Discover what makes Roameo your perfect travel companion
            </p>
          </div>

          <div className="mt-20">
            <RotatingFeatureDial features={featureHighlights} />
          </div>
        </div>
      </section>

      <LandingTripMoods />

      {/* Personalized Recommendations */}
      <section className="min-h-screen bg-[linear-gradient(180deg,#ffffff_0%,#f8fbff_26%,#edf6ff_62%,#dceeff_100%)] px-6 py-24 text-white">
        <div className="max-w-7xl mx-auto flex min-h-[70vh] items-center gap-16">
          <div className="flex-1">
            <div className="overflow-hidden rounded-[30px] border border-slate-100 bg-white p-5 shadow-[0_30px_80px_rgba(15,23,42,0.12)]">
              <div className="relative overflow-hidden rounded-[24px]">
                <img
                  src="/kerala-backwaters-houseboat.png"
                  alt="Luxury houseboat gliding through Kerala backwaters"
                  className="h-64 w-full object-cover"
                />
                <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(7,18,62,0.04)_0%,rgba(7,18,62,0.44)_100%)]" />
                <div className="absolute left-4 top-4 inline-flex items-center gap-2 rounded-full bg-white/88 px-3 py-2 text-sm font-medium text-slate-700 backdrop-blur-sm">
                  <MapPin className="h-4 w-4 text-sky-500" />
                  Alleppey, Kerala
                </div>
                <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
                  <div>
                    <h3 className="text-2xl font-semibold tracking-[-0.04em] text-white">Private Backwater Stay</h3>
                    <div className="mt-2 flex items-center gap-2 text-sm text-white/88">
                      <span>Slow water route</span>
                      <span className="text-white/56">•</span>
                      <span>Curated overnight stay</span>
                    </div>
                  </div>
                  <div className="rounded-full border border-white/18 bg-[#071238]/42 px-4 py-2 text-sm font-medium text-white backdrop-blur-md">
                    Sunset deck suite
                  </div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-[1.08fr_0.92fr] gap-3">
                <img
                  src="/kerala-houseboat.png"
                  alt="Interior of a Kerala houseboat suite"
                  className="h-[124px] w-full rounded-[18px] object-cover"
                />
                <div className="flex h-[124px] flex-col rounded-[18px] bg-[linear-gradient(145deg,#eef6ff_0%,#d8e8ff_100%)] px-4 py-3.5 text-slate-700">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Stay details</p>
                  <p className="mt-2 text-[0.96rem] font-semibold text-slate-900">2 travelers</p>
                  <p
                    className="mt-1.5 text-[0.78rem] leading-[1.15rem] text-slate-600"
                    style={{
                      display: "-webkit-box",
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}
                  >
                    2 nights on a private houseboat with dinner prepared on board.
                  </p>
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-3">
                <div className="rounded-[18px] border border-slate-200 bg-slate-50 px-4 py-3">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Check-in</p>
                  <p className="mt-2 text-lg font-semibold text-slate-900">Nov 08</p>
                </div>
                <div className="rounded-[18px] border border-slate-200 bg-slate-50 px-4 py-3">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Check-out</p>
                  <p className="mt-2 text-lg font-semibold text-slate-900">Nov 10</p>
                </div>
              </div>

              <div className="mt-5 flex items-center gap-3">
                <Button
                  onClick={() => handleProtectedAction("book now")}
                  className="flex-1 rounded-full bg-black text-white hover:bg-gray-800"
                >
                  Book now
                </Button>
                <Button
                  onClick={() => handleProtectedAction("add to trip")}
                  variant="outline"
                  className="rounded-full border-slate-200 px-5 text-slate-700"
                >
                  Add to trip
                </Button>
              </div>
            </div>
          </div>

          <div className="flex-1">
            <h2 className="text-4xl font-bold text-gray-900 mb-6 font-sans">Get personalized recommendations.</h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed font-sans">
              Roameo turns a rough travel brief into well-matched stays, scenic routes, and pacing decisions that
              already feel considered before you start editing the trip.
            </p>

            <div className="mb-6 flex items-start gap-3">
              <div className="flex h-10 w-14 items-center justify-center rounded-full bg-[rgba(53,115,255,1)]">
                <Compass className="w-5 h-5 text-white" />
              </div>
              <p className="pt-1 text-gray-700 font-sans">
                Save the options that fit, then keep route, timing, and places synced while you refine the plan.
              </p>
            </div>

            <Button
              onClick={() => handleProtectedAction("save to trip")}
              className="text-white hover:bg-sky-500 rounded-full px-8 bg-black"
            >
              Save to Trip
            </Button>
          </div>
        </div>
      </section>

      <footer className="relative overflow-hidden bg-[linear-gradient(180deg,#dceeff_0%,#cde5ff_100%)] px-6 pb-10 pt-12">
        <div className="absolute left-[-8%] top-14 h-56 w-56 rounded-full bg-white/65 blur-[70px]" />
        <div className="absolute right-[-6%] top-20 h-64 w-64 rounded-full bg-white/58 blur-[80px]" />

        <div className="mx-auto max-w-4xl text-center">
          <h2 className="mt-7 text-6xl font-semibold tracking-[-0.065em] text-[#101726]">
            Ready for the next trip?
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg leading-8 text-[#4b5f88]">
            Start in chat, refine on the map, and let every day, place, and timing decision stay connected while you plan.
          </p>

          <div className="mt-9 flex items-center justify-center gap-4">
            <Button
              onClick={() => handleProtectedAction("plan my trip now")}
              size="lg"
              className="rounded-full bg-black px-7 text-base text-white shadow-[0_14px_28px_rgba(0,0,0,0.16)] hover:bg-gray-800"
            >
              Start planning
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <a
              href="#features"
              className="inline-flex items-center rounded-full border border-[#b6cffd] bg-white/40 px-6 py-3 text-sm font-medium text-[#365dcb] transition-colors hover:bg-white/65"
            >
              Explore features
            </a>
          </div>
        </div>

        <div className="mx-auto mt-20 max-w-5xl rounded-[34px] border border-white/45 bg-white/38 px-8 py-10 shadow-[0_24px_60px_rgba(86,125,214,0.08)] backdrop-blur-md">
          <div className="grid gap-10 md:grid-cols-[1.3fr_0.85fr_0.85fr]">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-black">
                  <div className="h-2 w-2 rounded-full bg-white" />
                </div>
                <span className="text-2xl font-semibold tracking-[-0.05em] text-[#101726]">roameo</span>
              </div>
              <p className="mt-5 max-w-xs text-base leading-8 text-[#586b92]">
                Travel planning that keeps conversation, route, and day-by-day structure aligned from start to finish.
              </p>
            </div>

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6a82bb]">Pages</p>
              <div className="mt-5 space-y-4 text-sm text-[#25375f]">
                <a href="#how-it-works" className="block transition-colors hover:text-[#365dcb]">
                  How it works
                </a>
                <a href="#features" className="block transition-colors hover:text-[#365dcb]">
                  Features
                </a>
                <a href="/auth/login" className="block transition-colors hover:text-[#365dcb]">
                  Log in
                </a>
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6a82bb]">Information</p>
              <div className="mt-5 space-y-4 text-sm text-[#25375f]">
                <a href="#" className="block transition-colors hover:text-[#365dcb]">
                  Privacy
                </a>
                <a href="#" className="block transition-colors hover:text-[#365dcb]">
                  Terms
                </a>
                <a href="#" className="block transition-colors hover:text-[#365dcb]">
                  Support
                </a>
              </div>
            </div>
          </div>

          <div className="mt-10 border-t border-white/45 pt-6 text-sm text-[#66789f]">
            © 2026 Roameo. Thoughtful travel planning from first message to final day.
          </div>
        </div>
      </footer>
      </div>
    </div>
  )
}
