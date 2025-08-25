"use client"

import { Button } from "@/components/ui/button"
import { DirectionAwareHover } from "@/components/ui/direction-aware-hover"
import { MapPin, Star, Calendar } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import { supabase } from "@/lib/supabase/client"
import type { User } from "@supabase/supabase-js"

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

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-sky-300 via-blue-500 to-blue-800 px-6 min-h-screen flex items-center pt-20">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex-1 max-w-2xl">
            <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
              Plan your perfect
              <br />
              <span className="italic">adventure.</span>
            </h1>
            <p className="text-xl text-white/90 mb-8 leading-relaxed">
              Experience AI-powered travel planning that understands your preferences and creates personalized
              itineraries in minutes, not hours.
            </p>
            <div className="flex items-center gap-4">
              <Button
                onClick={() => handleProtectedAction("start planning")}
                size="lg"
                className="bg-black text-white hover:bg-gray-800 rounded-full px-8 py-3 text-lg"
              >
                Start planning
              </Button>
            </div>
          </div>

          <div className="flex-1 relative">
            <div className="absolute top-0 right-0 bg-white rounded-2xl p-6 shadow-2xl max-w-sm transform rotate-3">
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-700">
                      I'd love to help you plan your trip to Araku Valley! Are you interested in the coffee plantations,
                      tribal culture, or the scenic waterfalls?
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3 justify-end">
                  <div className="bg-blue-100 rounded-lg p-3 max-w-xs">
                    <p className="text-sm text-gray-700">
                      I want to experience the coffee plantation tours, visit the tribal museum, and see the Borra
                      Caves!
                    </p>
                  </div>
                  <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-700">
                      Perfect! I'm creating a 3-day Araku itinerary with visits to the Ananthagiri Coffee Plantations,
                      Tribal Museum, Borra Caves, and a scenic train journey through the Eastern Ghats...
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="px-6 min-h-screen flex items-center bg-slate-50">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-5xl font-bold text-center text-gray-900 mb-4 font-sans">How it Works</h2>
          <p className="text-xl text-gray-600 text-center mb-16 ">Three simple steps to your perfect trip</p>

          <div className="grid grid-cols-3 gap-12">
            <div className="text-center">
              <div className="w-16 h-16 bg-black rounded-full flex items-center justify-center mx-auto mb-6">
                <div className="w-4 h-4 bg-white rounded-full"></div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4 ">Chat with AI</h3>
              <p className="text-gray-600 leading-relaxed ">
                Tell our AI assistant about your dream destination in India, budget, interests, and travel style. The
                more details you share, the better we can personalize your experience.
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 bg-black">
                <Star className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4 font-sans">Get Smart Recommendations</h3>
              <p className="text-gray-600 leading-relaxed font-sans">
                Our AI analyzes thousands of travel data points to suggest accommodations, activities, restaurants, and
                hidden gems that match your preferences perfectly across India's diverse landscapes.
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 bg-black">
                <Calendar className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4 font-sans">Build Your Itinerary</h3>
              <p className="text-gray-600 leading-relaxed font-sans">
                Review, customize, and organize your personalized itinerary for India. Add or remove activities, adjust
                timing, and share with travel companions for the perfect trip.
              </p>
            </div>
          </div>
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

      {/* Popular Destinations */}
      <section className="px-6 py-24 bg-gray-50 min-h-screen flex items-center relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold text-gray-900 mb-6 font-sans">Popular Destinations</h2>
            <p className="text-xl text-gray-600 font-sans">
              Discover India's most breathtaking hill stations and coastal gems
            </p>
          </div>

          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              <DirectionAwareHover
                imageUrl="/munnar-tea-mist.png"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Munnar, Kerala</h3>
                  <p className="text-xs opacity-90">Misty tea gardens and rolling hills</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="/ooty-nilgiri-hills.png"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Ooty, Tamil Nadu</h3>
                  <p className="text-xs opacity-90">Queen of hill stations</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="/darjeeling-toy-train.png"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Darjeeling, West Bengal</h3>
                  <p className="text-xs opacity-90">Himalayan views and tea culture</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="/araku-valley-waterfalls.png"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Araku Valley</h3>
                  <p className="text-xs opacity-90">Coffee plantations and tribal culture</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="/rk-beach-visakhapatnam-golden-sand.png"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">RK Beach, Visakhapatnam</h3>
                  <p className="text-xs opacity-90">Golden sands and coastal charm</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="/kodaikanal-lake-mist.png"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Kodaikanal, Tamil Nadu</h3>
                  <p className="text-xs opacity-90">Princess of hill stations</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?q=80&w=2070&auto=format&fit=crop"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Shimla, Himachal Pradesh</h3>
                  <p className="text-xs opacity-90">Colonial charm in the mountains</p>
                </div>
              </DirectionAwareHover>

              <DirectionAwareHover
                imageUrl="https://images.unsplash.com/photo-1544735716-392fe2489ffa?q=80&w=2070&auto=format&fit=crop"
                className="w-full rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-white">
                  <h3 className="text-lg font-semibold mb-1">Manali, Himachal Pradesh</h3>
                  <p className="text-xs opacity-90">Adventure and serenity combined</p>
                </div>
              </DirectionAwareHover>
            </div>
          </div>
        </div>
      </section>

      {/* Call-to-Action Section */}
      <section className="px-6 bg-gradient-to-br from-sky-400 via-blue-600 to-blue-800 text-white py-20">
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

          <div className="flex items-center justify-center gap-6 mb-16">
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
  )
}
