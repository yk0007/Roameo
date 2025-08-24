"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { listTrips } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import { MapPin, Users, Send, LogOut, Plus, Mic, Clock, ArrowRight, Plane, Camera, Compass, Palmtree, Globe, Map } from "lucide-react"
import DestinationCardArt from "@/components/DestinationCardArt"

export default function Dashboard() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [message, setMessage] = useState("")
  const [loading, setLoading] = useState(true)
  const [trips, setTrips] = useState<Array<any>>([])

  // Load trips from backend (MVP: global list, no user filter yet)
  useEffect(() => {
    let mounted = true
    const load = () => {
      listTrips()
        .then(({ trips }) => {
          if (mounted) setTrips(trips)
        })
        .catch(() => {
          if (mounted) setTrips([])
        })
    }
    load()
    // Refresh when window regains focus or tab becomes visible (after deletions/navigation)
    const onFocus = () => load()
    const onVis = () => { if (document.visibilityState === "visible") load() }
    window.addEventListener("focus", onFocus)
    document.addEventListener("visibilitychange", onVis)
    return () => {
      mounted = false
      window.removeEventListener("focus", onFocus)
      document.removeEventListener("visibilitychange", onVis)
    }
  }, [])

  useEffect(() => {
    const getUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession()
      if (!session) {
        router.push("/auth/login")
        return
      }
      setUser(session.user)
      setLoading(false)
    }

    getUser()

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (!session) {
        router.push("/auth/login")
      } else {
        setUser(session.user)
      }
    })

    return () => subscription.unsubscribe()
  }, [router])

  const handleSignOut = async () => {
    // Navigate immediately for instant UX
    router.push("/auth/login")
    // Sign out in background
    await supabase.auth.signOut()
  }

  const handleSendMessage = () => {
    const text = message.trim()
    if (!text) return
    // Navigate to chat with initial message; chat page will send it and remove from URL
    router.push(`/chat?message=${encodeURIComponent(text)}`)
    setMessage("")
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading your dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen w-full bg-[#f8fafc] relative scroll-smooth">
      {/* Bottom Fade Grid Background */}
      <div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: `
            linear-gradient(to right, #e2e8f0 1px, transparent 1px),
            linear-gradient(to bottom, #e2e8f0 1px, transparent 1px)
          `,
          backgroundSize: "20px 30px",
          WebkitMaskImage: "radial-gradient(ellipse 70% 60% at 50% 100%, #000 60%, transparent 100%)",
          maskImage: "radial-gradient(ellipse 70% 60% at 50% 100%, #000 60%, transparent 100%)",
        }}
      />

      <div className="fixed inset-0 pointer-events-none z-0">
        <div
          className="absolute top-20 left-10 text-orange-600/60 animate-bounce"
          style={{ animationDelay: "0s", animationDuration: "3s" }}
        >
          <Plane className="w-10 h-10 rotate-45 drop-shadow-md" />
        </div>
        <div
          className="absolute top-32 right-16 text-amber-600/60 animate-bounce"
          style={{ animationDelay: "1s", animationDuration: "4s" }}
        >
          <Camera className="w-8 h-8 drop-shadow-md" />
        </div>
        <div
          className="absolute top-1/4 left-20 text-emerald-600/60 animate-bounce"
          style={{ animationDelay: "2s", animationDuration: "3.5s" }}
        >
          <Compass className="w-9 h-9 drop-shadow-md" />
        </div>
        <div
          className="absolute top-1/3 right-10 text-green-600/60 animate-bounce"
          style={{ animationDelay: "0.5s", animationDuration: "4.5s" }}
        >
          <Palmtree className="w-10 h-10 drop-shadow-md" />
        </div>
        <div
          className="absolute bottom-1/3 left-16 text-teal-600/60 animate-bounce"
          style={{ animationDelay: "1.5s", animationDuration: "3.8s" }}
        >
          <Globe className="w-8 h-8 drop-shadow-md" />
        </div>
        <div
          className="absolute bottom-1/4 right-20 text-red-600/60 animate-bounce"
          style={{ animationDelay: "2.5s", animationDuration: "4.2s" }}
        >
          <Map className="w-9 h-9 drop-shadow-md" />
        </div>
        <div
          className="absolute top-1/2 left-8 text-yellow-600/60 animate-bounce"
          style={{ animationDelay: "3s", animationDuration: "3.3s" }}
        >
          <Plane className="w-7 h-7 rotate-12 drop-shadow-md" />
        </div>
        <div
          className="absolute bottom-20 right-8 text-orange-500/60 animate-bounce"
          style={{ animationDelay: "1.8s", animationDuration: "4.8s" }}
        >
          <Camera className="w-8 h-8 rotate-45 drop-shadow-md" />
        </div>
      </div>

      <header className="flex items-center justify-between px-6 py-3 border-0 bg-white/80 backdrop-blur-md shadow-lg">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center shadow-md">
                <div className="w-2 h-2 bg-white rounded-full"></div>
              </div>
              <span className="text-xl font-bold text-gray-900">roameo</span>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-muted-foreground">
            Welcome, {user?.user_metadata?.username || user?.user_metadata?.first_name || user?.email?.split("@")[0]}
          </span>
          <Button
            onClick={() => router.push("/profile")}
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
          >
            Profile
          </Button>
          <Button
            onClick={handleSignOut}
            variant="outline"
            size="sm"
            className="flex items-center space-x-2 rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
          >
            <LogOut className="h-4 w-4" />
            <span>Sign Out</span>
          </Button>
        </div>
      </header>

      <div className="relative overflow-hidden z-10 min-h-screen flex items-center justify-center">
        <div className="relative max-w-4xl mx-auto px-6 text-center">
          <div className="space-y-6">
            <p className="text-sm text-slate-600 font-medium tracking-wide uppercase opacity-0 animate-fade-in-up">
              I'm
            </p>
            <h1 className="text-7xl md:text-8xl font-bold text-slate-900 tracking-tight opacity-0 animate-fade-in-scale animation-delay-200">
              roameo
            </h1>
            <p className="text-xl md:text-2xl text-slate-600 max-w-2xl mx-auto leading-relaxed opacity-0 animate-fade-in-up animation-delay-400">
              Here to create a master plan for your dream trip loaded with memories.
            </p>
          </div>

          <div className="mt-12 opacity-0 animate-fade-in-up animation-delay-600">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleSendMessage()
              }}
              className="relative"
            >
              <div className="bg-white/80 backdrop-blur-md border rounded-3xl shadow-lg p-4 border-zinc-300">
                <div className="flex items-center gap-3 flex-row">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="flex-shrink-0 w-10 h-10 rounded-full hover:bg-white/80"
                  >
                    <Plus className="w-5 h-5" />
                  </Button>

                  <Input
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Where would you like to go? What kind of experience are you looking for?"
                    className="flex-1 border-0 bg-transparent text-lg placeholder:text-gray-500 focus-visible:ring-0 focus-visible:ring-offset-0 px-0 shadow-none"
                  />

                  <div className="flex items-center gap-2 flex-shrink-0">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="w-10 h-10 rounded-full hover:bg-white/80"
                    >
                      <Mic className="w-5 h-5" />
                    </Button>
                    <Button
                      type="submit"
                      variant="ghost"
                      size="sm"
                      className="w-10 h-10 rounded-full hover:bg-white/80"
                      disabled={!message.trim()}
                    >
                      <Send className="w-5 h-5" />
                    </Button>
                  </div>
                </div>
              </div>
            </form>

            <p className="text-xs text-gray-500 text-center mt-3">Roameo can make mistakes. Check important info.</p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 relative z-10">
        <div>
          <div className="flex items-center justify-between mb-8">
            <div className="opacity-0 animate-slide-in-left animation-delay-800">
              <h2 className="text-3xl font-bold text-slate-900 mb-2">Your Trips</h2>
              <p className="text-slate-600">Manage and explore your planned adventures</p>
            </div>
            <Button className="bg-black hover:bg-gray-800 text-white rounded-full px-6 opacity-0 animate-fade-in-up animation-delay-800" onClick={() => router.push("/chat")}>
              <Plus className="w-4 h-4 mr-2" />
              Plan New Trip
            </Button>
          </div>

          {trips.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {trips.map((trip: any, index: number) => {
                // Color palette for the dots
                const colors = [
                  'bg-blue-500',
                  'bg-red-500', 
                  'bg-green-500',
                  'bg-gray-500',
                  'bg-orange-500',
                  'bg-purple-500',
                  'bg-yellow-500',
                  'bg-pink-500'
                ]
                const colorClass = colors[index % colors.length]
                
                return (
                  <Card 
                    key={trip.id} 
                    className="group hover:shadow-lg transition-all duration-300 border border-gray-200/50 overflow-hidden bg-white rounded-3xl hover:-translate-y-1 cursor-pointer"
                    onClick={() => router.push(`/chat?sessionId=${encodeURIComponent(trip.id)}`)}
                  >
                    <div className="p-6">
                      {/* Destination card art */}
                      <div className="relative h-48 bg-gray-100 rounded-2xl mb-4 overflow-hidden">
                        <DestinationCardArt
                          destination={trip.destination || trip.title}
                          variant="stamp"
                          className="rounded-2xl"
                        />
                      </div>
                      
                      {/* Content */}
                      <div className="space-y-3">
                        {/* Title with colored dot */}
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${colorClass} flex-shrink-0`}></div>
                          <h3 className="font-semibold text-gray-900 text-lg truncate">{trip.title}</h3>
                        </div>
                        
                        {/* Destination with subtle styling */}
                        <div className="flex items-center gap-2 text-gray-500 text-sm">
                          <MapPin className="w-4 h-4" />
                          <span>{trip.destination}</span>
                        </div>
                      </div>
                    </div>
                  </Card>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-16 border rounded-2xl bg-white/70 backdrop-blur-sm">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
                <MapPin className="w-8 h-8 text-gray-500" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">No trips yet</h3>
              <p className="text-slate-600 mb-6">Start planning your first adventure with Roameo</p>
              <Button className="bg-black hover:bg-gray-800 text-white rounded-full px-6" onClick={() => router.push("/chat") }>
                <Plus className="w-4 h-4 mr-2" /> Plan New Trip
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
