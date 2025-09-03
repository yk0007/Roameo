"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { CachedImage } from "@/components/cached-image"
import { Plus, MapPin, Calendar, Users, Plane, Camera, Mountain, Send } from "lucide-react"
import { supabase } from "@/lib/supabase/client"

export function Dashboard() {
  const [message, setMessage] = useState("")
  const [trips, setTrips] = useState<Array<any>>([])
  const [loading, setLoading] = useState(true)
  const [imageLoadingStates, setImageLoadingStates] = useState<Record<string, boolean>>({})
  const [heroVisible, setHeroVisible] = useState(false)
  const [cardsVisible, setCardsVisible] = useState(false)
  const [chatVisible, setChatVisible] = useState(false)
  const [typingText, setTypingText] = useState("")
  const fullText = "Here to create a master plan for your dream trip loaded with memories."

  // Text typing animation effect
  useEffect(() => {
    setHeroVisible(true)
    let currentIndex = 0
    const typingTimer = setInterval(() => {
      if (currentIndex <= fullText.length) {
        setTypingText(fullText.slice(0, currentIndex))
        currentIndex++
      } else {
        clearInterval(typingTimer)
      }
    }, 50) // Adjust speed here

    // Stagger other animations
    const chatTimer = setTimeout(() => setChatVisible(true), 1000)
    const cardsTimer = setTimeout(() => setCardsVisible(true), 1500)

    return () => {
      clearInterval(typingTimer)
      clearTimeout(chatTimer)
      clearTimeout(cardsTimer)
    }
  }, [])

  useEffect(() => {
    const fetchTrips = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession()
        if (!session) {
          // Redirect to auth page immediately if no session
          window.location.href = '/auth/login'
          return
        }

        const response = await fetch('/api/trips/list', {
          headers: {
            'Authorization': `Bearer ${session.access_token}`,
            'Cache-Control': 'no-cache'
          }
        })

        if (response.ok) {
          const data = await response.json()
          console.log("Trips API response:", data)
          setTrips(data.trips || [])
        } else {
          console.error("Trips API error:", response.status, await response.text())
          // Don't show error to user, just log it
          setTrips([])
        }
      } catch (error) {
        console.error('Error fetching trips:', error)
        setTrips([])
      } finally {
        setLoading(false)
      }
    }

    fetchTrips()
  }, [])

  // Function to generate a fallback image URL for destinations
  const getFallbackImageUrl = (destination: string) => {
    if (!destination) return null
    // Use a placeholder service or generate a themed background
    return `https://source.unsplash.com/800x600/?${encodeURIComponent(destination)},travel,landscape`
  }

  // Function to get the best available image for a trip
  const getTripImageUrl = (trip: any) => {
    if (trip.destinationImageUrl) {
      return trip.destinationImageUrl
    }
    if (trip.image) {
      return trip.image
    }
    if (trip.destination) {
      return getFallbackImageUrl(trip.destination)
    }
    return null
  }

  const handleSendMessage = () => {
    if (message.trim()) {
      // Redirect to chat page with the message
      window.location.href = `/chat?message=${encodeURIComponent(message)}`
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile Notice - Only visible on small screens */}
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white p-6 md:hidden">
        <div className="text-center max-w-sm animate-fade-in">
          <div className="w-16 h-16 bg-black rounded-full flex items-center justify-center mx-auto mb-6 animate-bounce">
            <div className="w-4 h-4 bg-white rounded-full"></div>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-4">roameo</h1>
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Desktop Experience Required</h2>
          <p className="text-gray-600 mb-6 leading-relaxed">
            For the best experience, please visit our website on a desktop or laptop computer.
          </p>
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-4 animate-pulse">
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
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-secondary/5" />
        <div className="relative max-w-4xl mx-auto px-6 py-16 text-center">
          <div className={`space-y-4 transition-all duration-1000 transform ${
            heroVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
          }`}>
            <p className="text-sm text-muted-foreground font-medium tracking-wide uppercase">I'm</p>
            <h1 className="text-6xl md:text-7xl font-bold text-foreground tracking-tight bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
              Roameo
            </h1>
            <div className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed min-h-[3rem] flex items-center justify-center">
              <span className="border-r-2 border-primary pr-1">{typingText}</span>
            </div>
          </div>

          {/* Decorative elements static */}
          <div className="absolute top-20 left-10 text-primary/20">
            <Plane className="w-8 h-8" />
          </div>
          <div className="absolute top-32 right-16 text-secondary/20">
            <Camera className="w-6 h-6" />
          </div>
          <div className="absolute bottom-20 left-20 text-primary/20">
            <Mountain className="w-10 h-10" />
          </div>
        </div>
      </div>

      {/* Chat Interface */}
      <div className={`max-w-4xl mx-auto px-6 py-8 transition-all duration-800 transform ${
        chatVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
      }`}>
        <Card className="border-border/50 shadow-lg hover:shadow-2xl transition-all duration-500 hover:scale-[1.02] group">
          <CardContent className="p-6">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                  <MapPin className="w-5 h-5 text-primary group-hover:scale-110 transition-transform duration-300" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors duration-300">Start Planning Your Adventure</h3>
                  <p className="text-sm text-muted-foreground">Tell me about your dream destination</p>
                </div>
              </div>

              <div className="relative">
                <Textarea
                  placeholder="Where would you like to go? What kind of experience are you looking for?"
                  value={message}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setMessage(e.target.value)}
                  className="min-h-[120px] resize-none border-border/50 focus:border-primary/50 focus:ring-primary/20 transition-all duration-300 focus:scale-[1.01]"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!message.trim()}
                  className="absolute bottom-3 right-3 h-8 w-8 p-0 hover:scale-110 transition-transform duration-300"
                  size="sm"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Your Trips Section */}
      <div className={`max-w-6xl mx-auto px-6 py-12 transition-all duration-1000 transform ${
        cardsVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
      }`}>
        <div className="space-y-8">
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-bold text-foreground bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Your Trips</h2>
            <p className="text-muted-foreground">Manage and explore your planned adventures</p>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center animate-pulse">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
              <p className="text-muted-foreground animate-pulse">Loading your trips...</p>
            </div>
          ) : trips.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {trips.map((trip: any, index: number) => {
                const imageUrl = getTripImageUrl(trip)
                const isVisible = index < 6 // Only load first 6 images immediately
                
                return (
                <Card 
                  key={trip.id} 
                  className={`group hover:shadow-2xl transition-all duration-500 border-border/50 overflow-hidden cursor-pointer rounded-2xl transform hover:scale-[1.03] hover:-translate-y-2 ${
                    cardsVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
                  }`}
                  style={{ 
                    animationDelay: `${index * 100}ms`,
                    transitionDelay: `${index * 100}ms`
                  }}
                  onClick={() => window.location.href = `/chat?sessionId=${trip.sessionId}`}
                >
                  <div className="relative h-48 bg-gradient-to-br from-blue-500 to-purple-600 overflow-hidden rounded-2xl mx-4 mt-4 group-hover:shadow-lg transition-all duration-500">
                    {imageUrl ? (
                      <CachedImage 
                        src={imageUrl} 
                        alt={trip.destination || trip.title || 'Trip destination'}
                        className="group-hover:scale-125 transition-transform duration-700 ease-out" 
                        priority={isVisible}
                        quality={isVisible ? 90 : 75}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-white relative group-hover:scale-105 transition-transform duration-500">
                        {/* Colorful gradient background with destination initial */}
                        <div className="absolute inset-0 bg-gradient-to-br from-pink-500 via-purple-500 to-indigo-600 group-hover:from-pink-600 group-hover:via-purple-700 group-hover:to-indigo-800 transition-all duration-500"></div>
                        <div className="relative z-10 text-center">
                          <div className="text-6xl font-bold mb-2 opacity-90 group-hover:scale-110 transition-transform duration-300">
                            {(trip.destination || trip.title || 'T').charAt(0).toUpperCase()}
                          </div>
                          <p className="text-sm font-medium opacity-80">{trip.destination || trip.title}</p>
                        </div>
                        {/* Decorative elements */}
                        <div className="absolute top-4 right-4 opacity-20 group-hover:opacity-40 transition-opacity duration-300">
                          <MapPin className="w-8 h-8 group-hover:scale-110 transition-transform duration-300" />
                        </div>
                      </div>
                    )}
                    {/* Overlay gradient on hover */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  </div>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg font-bold text-foreground group-hover:text-primary transition-colors duration-300">{trip.title}</CardTitle>
                    <CardDescription className="text-muted-foreground group-hover:text-foreground/80 transition-colors duration-300">{trip.destination}</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <div className="flex items-center gap-1 group-hover:text-primary transition-colors duration-300">
                        <Calendar className="w-4 h-4 group-hover:scale-110 transition-transform duration-300" />
                        <span>{trip.duration || 'Duration not set'}</span>
                      </div>
                      <div className="flex items-center gap-1 group-hover:text-primary transition-colors duration-300">
                        <Users className="w-4 h-4 group-hover:scale-110 transition-transform duration-300" />
                        <span>{trip.travelers || '1 traveler'}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-12 animate-fade-in">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center hover:scale-110 transition-transform duration-300 animate-bounce">
                <MapPin className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">No trips yet</h3>
              <p className="text-muted-foreground mb-4">Start planning your first adventure with Roameo</p>
              <Button className="bg-primary hover:bg-primary/90 hover:scale-105 transition-all duration-300 group">
                <Plus className="w-4 h-4 mr-2 group-hover:rotate-90 transition-transform duration-300" />
                Plan New Trip
              </Button>
            </div>
          )}
        </div>
      </div>
      </div>
    </div>
  )
}
