"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { CachedImage } from "@/components/cached-image"
import { Plus, MapPin, Calendar, Users, Plane, Camera, Mountain, Send } from "lucide-react"

export function Dashboard() {
  const [message, setMessage] = useState("")
  const [trips] = useState<Array<any>>([]) // TODO: load user trips when backend is ready

  const handleSendMessage = () => {
    if (message.trim()) {
      // Handle message sending logic here
      console.log("Sending message:", message)
      setMessage("")
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-secondary/5" />
        <div className="relative max-w-4xl mx-auto px-6 py-16 text-center">
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground font-medium tracking-wide uppercase">I'm</p>
            <h1 className="text-6xl md:text-7xl font-bold text-foreground tracking-tight">Roameo</h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Here to create a master plan for your dream trip loaded with memories.
            </p>
          </div>

          {/* Decorative elements */}
          <div className="absolute top-20 left-10 text-primary/20">
            <Plane className="w-8 h-8 rotate-45" />
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
      <div className="max-w-4xl mx-auto px-6 py-8">
        <Card className="border-border/50 shadow-lg">
          <CardContent className="p-6">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <MapPin className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Start Planning Your Adventure</h3>
                  <p className="text-sm text-muted-foreground">Tell me about your dream destination</p>
                </div>
              </div>

              <div className="relative">
                <Textarea
                  placeholder="Where would you like to go? What kind of experience are you looking for?"
                  value={message}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setMessage(e.target.value)}
                  className="min-h-[120px] resize-none border-border/50 focus:border-primary/50 focus:ring-primary/20"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!message.trim()}
                  className="absolute bottom-3 right-3 h-8 w-8 p-0"
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
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="space-y-8">
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-bold text-foreground">Your Trips</h2>
            <p className="text-muted-foreground">Manage and explore your planned adventures</p>
          </div>

          {trips.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {trips.map((trip: any) => (
                <Card key={trip.id} className="group hover:shadow-xl transition-all duration-300 border-border/50 overflow-hidden">
                  <div className="relative bg-gradient-to-br from-blue-500 to-purple-600">
                    {trip.image ? (
                      <CachedImage 
                        src={trip.image} 
                        alt={trip.title} 
                        className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300" 
                      />
                    ) : (
                      <div className="w-full h-48 flex items-center justify-center text-white">
                        <div className="text-center">
                          <MapPin className="w-12 h-12 mx-auto mb-2 opacity-80" />
                          <p className="text-sm font-medium">{trip.destination || trip.title}</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg font-bold text-foreground group-hover:text-primary transition-colors">{trip.title}</CardTitle>
                    <CardDescription className="text-muted-foreground">{trip.destination}</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Calendar className="w-4 h-4" />
                        <span>{trip.duration}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Users className="w-4 h-4" />
                        <span>{trip.travelers}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <MapPin className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">No trips planned yet</h3>
              <p className="text-muted-foreground mb-4">Start planning your first adventure with Roameo</p>
              <Button className="bg-primary hover:bg-primary/90" onClick={() => (window.location.href = "/chat")}>Plan Your First Trip</Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
