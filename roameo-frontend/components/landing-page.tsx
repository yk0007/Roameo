"use client"

import { Button } from "@/components/ui/button"
import { DirectionAwareHover } from "@/components/ui/direction-aware-hover"
import { MapPin, Star, Calendar } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import { supabase } from "@/lib/supabase/client"
import type { User } from "@supabase/supabase-js"
import { FeaturesSectionWithHoverEffects } from "@/components/blocks/feature-section-with-hover-effects"
import { BlurFade } from "@/components/ui/blur-fade"
import { motion } from "framer-motion"
import HeroScrollAnimation from "@/components/ui/hero-scroll-animation"

export function LandingPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)

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
    } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user || null)
    })

    return () => subscription.unsubscribe()
  }, [])

  const handleProtectedAction = (action: string) => {
    if (!user) {
      router.push("/auth/login")
      return
    }
    router.push("/dashboard")
  }

  const handleSignOut = async () => {
    // Navigate immediately for instant UX
    router.push("/auth/login")
    // Sign out in background
    await supabase.auth.signOut()
    setUser(null)
  }

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
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 px-6 py-4 flex items-center justify-between bg-white/80 backdrop-blur-md border-b border-white/30 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
            <div className="w-2 h-2 bg-white rounded-full"></div>
          </div>
          <span className="text-xl font-bold text-gray-900">roameo</span>
        </div>
        
        <div className="flex items-center gap-6">
          {user ? (
            <Button onClick={handleSignOut} variant="ghost" className="text-gray-600">
              Sign out
            </Button>
          ) : (
            <Link href="/auth/login">
              <Button variant="ghost" className="text-gray-600">
                Log in
              </Button>
            </Link>
          )}
          <Button
            onClick={() => handleProtectedAction("get started")}
            className="bg-black text-white hover:bg-gray-800 rounded-full px-6"
          >
            Get started
          </Button>
        </div>
      </header>

      {/* Hero Section with Scroll Animation */}
      <HeroScrollAnimation user={user} handleProtectedAction={handleProtectedAction} />

      {/* How it Works Section */}
      <section className="px-6 min-h-screen flex items-center bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-900 mb-4 font-sans">How it Works</h2>
            <p className="text-xl text-gray-600">Discover the power of AI-driven travel planning</p>
          </div>
          <FeaturesSectionWithHoverEffects />
        </div>
      </section>

      {/* Features Section */}
      <section className="px-6 bg-white min-h-screen flex items-center">
        <div className="max-w-7xl mx-auto rounded-3xl border-0 shadow-xl px-10 py-10">
          {/* Figma-style colorful logo */}

          <h2 className="text-4xl font-bold text-gray-900 mb-4 font-sans">Features</h2>
          <p className="text-lg text-gray-600 mb-12 font-sans">
            Discover what makes Roameo your perfect travel companion
          </p>

          {/* Features grid with geometric icons */}
          <div className="bg-white rounded-3xl p-12 max-w-5xl mx-auto shadow-none">
            <div className="grid grid-cols-3 gap-12">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-blue-400 to-purple-600 flex items-center justify-center shadow-lg">
                  <div className="relative">
                    <div className="w-8 h-8 border-2 border-white rounded-full"></div>
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-1 h-3 bg-white rounded-full"></div>
                    <div className="absolute top-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-2 border-r-2 border-b-2 border-transparent border-b-white"></div>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3 font-sans">Local Expertise</h3>
                <p className="text-gray-600 text-sm leading-relaxed font-sans">
                  Navigate India's diverse regions with insider knowledge. Get authentic recommendations from local
                  experts across hill stations and coastal areas.
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-emerald-400 to-green-600 flex items-center justify-center shadow-lg rounded-lg">
                  <div className="relative">
                    <div className="w-0 h-0 border-l-3 border-r-3 border-b-4 border-transparent border-b-white"></div>
                    <div className="absolute -top-1 left-1 w-0 h-0 border-l-2 border-r-2 border-b-3 border-transparent border-b-white"></div>
                    <div className="absolute -top-0.5 -left-1 w-0 h-0 border-l-2 border-r-2 border-b-2 border-transparent border-b-white"></div>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3 font-sans">Hidden Gems</h3>
                <p className="text-gray-600 text-sm leading-relaxed font-sans">
                  Discover secret viewpoints in Ooty, untouched waterfalls in Araku, and local festivals that typical
                  guides miss.
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg rounded-lg">
                  <div className="relative">
                    <div className="w-8 h-6 bg-white rounded-sm"></div>
                    <div className="absolute top-1 left-1 w-6 h-1 bg-cyan-600 rounded-full"></div>
                    <div className="absolute top-3 left-1 w-4 h-1 bg-cyan-600 rounded-full"></div>
                    <div className="absolute top-1 right-1 w-2 h-2 bg-cyan-600 rounded-full"></div>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3 font-sans">Smart Budgeting</h3>
                <p className="text-gray-600 text-sm leading-relaxed font-sans">
                  From budget stays in Darjeeling to luxury resorts in Munnar, get cost-effective recommendations that
                  maximize value.
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-rose-400 to-pink-600 flex items-center justify-center shadow-lg rounded-lg">
                  <div className="relative">
                    <div className="w-6 h-8 bg-white rounded-t-full"></div>
                    <div className="absolute bottom-0 left-0 w-8 h-2 bg-white rounded-sm"></div>
                    <div className="absolute top-2 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-rose-600 rounded-full"></div>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3 font-sans">Cultural Sensitivity</h3>
                <p className="text-gray-600 text-sm leading-relaxed font-sans">
                  Navigate India's rich cultural landscape with confidence. Get guidance on local customs and regional
                  traditions.
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-amber-400 to-orange-600 flex items-center justify-center shadow-lg rounded-lg">
                  <div className="relative">
                    <div className="w-8 h-6 bg-white rounded-sm"></div>
                    <div className="absolute top-0 left-0 w-8 h-2 bg-orange-600 rounded-t-sm"></div>
                    <div className="grid grid-cols-4 gap-0.5 absolute top-2.5 left-1 w-6 h-3">
                      <div className="w-1 h-1 bg-orange-600 rounded-full"></div>
                      <div className="w-1 h-1 bg-orange-600 rounded-full"></div>
                      <div className="w-1 h-1 bg-orange-600 rounded-full"></div>
                      <div className="w-1 h-1 bg-orange-600 rounded-full"></div>
                    </div>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3 font-sans">Seasonal Planning</h3>
                <p className="text-gray-600 text-sm leading-relaxed font-sans">
                  Plan around monsoons and peak seasons. Visit Munnar during tea harvest or RK Beach during pleasant
                  winters.
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-green-400 to-emerald-500 to-teal-800 opacity-95 rounded-full"></div>
                <h3 className="text-xl font-bold text-gray-900 mb-3 font-sans">Real-time Updates</h3>
                <p className="text-gray-600 text-sm leading-relaxed font-sans">
                  Stay informed about weather conditions in hill stations, local events, and travel advisories for
                  smooth journeys.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>



      {/* Personalized Recommendations */}
      <section className="px-6 min-h-screen flex items-center text-white bg-white border">
        <div className="max-w-7xl mx-auto flex items-center gap-16">
          <div className="flex-1">
            <div className="bg-white p-8 rounded-4xl shadow-xl rounded-3xl px-9 py-0">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-sky-400 rounded-full flex items-center justify-center">
                  <MapPin className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-bold text-lg text-black">Tea Valley Resort, Munnar</h3>
                  <div className="flex items-center gap-1">
                    <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                    <span className="text-sm text-gray-600">4.7 • Hill Station Resort</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-6">
                <img
                  src="/munnar-tea-gardens-sunrise.png"
                  alt="Munnar tea gardens"
                  className="h-32 w-full object-cover rounded-lg"
                />
                <img
                  src="/munnar-tea-hills.png"
                  alt="Munnar misty hills"
                  className="h-32 w-full object-cover rounded-lg"
                />
                <img
                  src="/luxury-resort-room.png"
                  alt="Tea valley resort room"
                  className="h-20 w-full object-cover rounded-lg"
                />
                <div className="h-20 bg-gradient-to-br from-gray-400 to-gray-500 rounded-lg flex items-center justify-center">
                  <span className="text-white text-sm font-medium">Show all photos</span>
                </div>
              </div>

              <div className="flex items-center justify-between mb-6 text-sm text-gray-600">
                <div>
                  <p className="font-medium">Check-in</p>
                  <p>Oct 12</p>
                </div>
                <div>
                  <p className="font-medium">Check-out</p>
                  <p>Oct 15</p>
                </div>
              </div>

              <p className="text-sm text-gray-600 mb-6">2 Travelers, 1 Valley View Room</p>

              <div className="space-y-3">
                <Button
                  onClick={() => handleProtectedAction("book now")}
                  className="w-full bg-black text-white hover:bg-gray-800 rounded-full"
                >
                  Book now
                </Button>
                <Button
                  onClick={() => handleProtectedAction("add to trip")}
                  variant="outline"
                  className="w-full rounded-full bg-transparent"
                >
                  + Add to trip
                </Button>
              </div>
            </div>
          </div>

          <div className="flex-1">
            <h2 className="text-4xl font-bold text-gray-900 mb-6 font-sans">Get personalized recommendations.</h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed font-sans">
              Discover India's breathtaking hill stations and coastal gems through AI-powered recommendations. From
              misty tea gardens in Munnar to serene beaches at RK Beach, we'll curate experiences perfect for your
              journey.
            </p>

            <div className="flex items-center gap-3 p-4 bg-sky-50 rounded-lg mb-6">
              <div className="h-10 rounded-full flex items-center justify-center w-14 bg-[rgba(53,115,255,1)]">
                <Star className="w-5 h-5 text-white" />
              </div>
              <p className="text-gray-700 font-sans">
                Save your favorite places and build the perfect itinerary for your Indian adventure
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

      {/* Call-to-Action Section */}
      <section className="px-6 bg-gradient-to-br from-sky-400 via-blue-600 to-blue-800 text-white py-12">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl font-bold mb-6 leading-tight font-sans">
            Ready for Your Next
            <br />
            Adventure?
          </h2>
          <p className="text-xl text-white/90 mb-12 leading-relaxed max-w-3xl mx-auto font-sans">
            Join thousands of travelers who've discovered the future of trip planning. Start creating your perfect
            itinerary today.
          </p>

          <div className="flex items-center justify-center gap-6 mb-8">
            <Button
              onClick={() => handleProtectedAction("plan my trip now")}
              size="lg"
              className="bg-white/20 backdrop-blur-sm text-white border border-white/30 hover:bg-white/30 rounded-full px-8 py-4 text-lg flex items-center gap-3"
            >
              <Calendar className="w-5 h-5" />
              Plan My Trip Now
            </Button>
            <Button variant="ghost" size="lg" className="text-white hover:bg-white/10 rounded-full px-8 py-4 text-lg">
              Learn More
            </Button>
          </div>

          <div className="flex items-center justify-center gap-12 text-white/90">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center">
                <div className="w-3 h-3 text-white">✓</div>
              </div>
              <span className="font-sans">Free to start</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center">
                <div className="w-3 h-3 text-white">✓</div>
              </div>
              <span className="font-sans">No credit card required</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center">
                <div className="w-3 h-3 text-white">✓</div>
              </div>
              <span className="font-sans">Instant results</span>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 bg-gray-50 border-t border-gray-200">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 bg-black rounded-full flex items-center justify-center">
              <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
            </div>
            <span className="font-bold text-gray-900">roameo</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-gray-600">
            <a href="#" className="hover:text-gray-900 transition-colors">
              Privacy
            </a>
            <a href="#" className="hover:text-gray-900 transition-colors">
              Terms
            </a>
            <a href="#" className="hover:text-gray-900 transition-colors">
              Support
            </a>
          </div>
        </div>
      </footer>
      </div>
    </div>
  )
}
