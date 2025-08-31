"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { supabase } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { listTrips } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { CachedImage } from "@/components/cached-image"
import { ArrowRight, User, LogOut, Github, Heart, ExternalLink } from "lucide-react"

export default function Dashboard() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [trips, setTrips] = useState<Array<any>>([])
  const [planningForm, setPlanningForm] = useState({
    destination: '',
    days: '',
    budget: '',
    travellers: 'nature' // Set Nature as default
  })
  const [quickMessage, setQuickMessage] = useState('')
  const [activeInput, setActiveInput] = useState('none') // 'form', 'chat', or 'none'
  
  // Animation states - only enable if not coming from chat
  const [isFromChat, setIsFromChat] = useState(false)
  const [heroVisible, setHeroVisible] = useState(false)
  const [formVisible, setFormVisible] = useState(false)
  const [tripsVisible, setTripsVisible] = useState(false)
  const fullText = "Here to add memories to your life"

  // Animation effect - only if not from chat
  useEffect(() => {
    // Check if coming from chat page using multiple methods
    const fromChatReferrer = document.referrer.includes('/chat')
    const fromChatState = window.history.state?.fromChat
    const fromChatSession = sessionStorage.getItem('fromChat') === 'true'
    const fromChat = fromChatReferrer || fromChatState || fromChatSession
    
    setIsFromChat(fromChat)
    
    // Clear the flag after checking
    if (fromChatSession) {
      sessionStorage.removeItem('fromChat')
    }
    
    if (fromChat) {
      // Skip animations and show everything immediately
      setHeroVisible(true)
      setFormVisible(true)
      setTripsVisible(true)
      return
    }
    
    // Normal animation sequence
    setHeroVisible(true)

    // Stagger other animations
    const formTimer = setTimeout(() => setFormVisible(true), 1000)
    const tripsTimer = setTimeout(() => setTripsVisible(true), 1500)

    return () => {
      clearTimeout(formTimer)
      clearTimeout(tripsTimer)
    }
  }, [])



  // Load trips from backend with proper authentication
  useEffect(() => {
    let mounted = true
    let isLoading = false
    
    const load = async () => {
      if (isLoading || !user) {
        console.log('Skipping load:', { isLoading, hasUser: !!user })
        return
      }
      isLoading = true
      console.log('Loading trips for user:', user?.email)
      
      try {
        const { trips } = await listTrips()
        console.log('API response received:', { trips: trips, tripsLength: trips?.length })
        if (mounted) {
          // Always use real data from backend, even if empty
          setTrips(trips || [])
          console.log('Set trips state:', trips?.length || 0, 'trips')
          console.log('First trip example:', trips?.[0])
        }
      } catch (error) {
        console.error('Failed to load trips:', error)
        if (mounted) {
          // Show empty state when there's an error
          console.log('Setting empty trips array due to API error')
          setTrips([])
        }
      } finally {
        isLoading = false
      }
    }
    
    if (user) {
      load()
    }
    
    return () => {
      mounted = false
    }
  }, [user])

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

  const handlePlanTrip = () => {
    // Validate required fields
    if (!planningForm.destination.trim()) {
      alert('Please enter a destination')
      return
    }
    if (!planningForm.days.trim()) {
      alert('Please enter number of days')
      return
    }
    
    // Create a dynamic prompt based on available inputs
    const tripType = planningForm.travellers || 'nature'
    const destination = planningForm.destination.trim()
    const days = planningForm.days.trim()
    const budget = planningForm.budget.trim()
    
    let prompt = `Plan a ${tripType} trip to ${destination} for ${days} days`
    
    // Add budget information if provided
    if (budget) {
      prompt += ` with a budget of INR ${budget}`
    }
    
    // Add specific requirements based on trip type
    if (tripType === 'luxury') {
      prompt += `. Focus on premium accommodations, fine dining, and exclusive experiences`
    } else if (tripType === 'budget') {
      prompt += `. Focus on affordable accommodations, local transportation, and budget-friendly activities`
    } else if (tripType === 'adventure') {
      prompt += `. Include thrilling activities, outdoor adventures, and adrenaline-pumping experiences`
    } else if (tripType === 'family') {
      prompt += `. Include family-friendly activities, suitable accommodations, and child-safe attractions`
    } else if (tripType === 'romantic') {
      prompt += `. Focus on romantic experiences, intimate settings, and couple-friendly activities`
    } else if (tripType === 'cultural') {
      prompt += `. Emphasize cultural sites, historical landmarks, local traditions, and authentic experiences`
    } else if (tripType === 'nature') {
      prompt += `. Focus on natural attractions, wildlife, scenic landscapes, and eco-friendly activities`
    } else if (tripType === 'business') {
      prompt += `. Include business-friendly accommodations, meeting facilities, and professional services`
    }
    
    // Add general requirements
    prompt += `. Please provide a detailed itinerary with:`
    prompt += `\n- Day-by-day schedule with specific activities`
    prompt += `\n- Recommended places to visit with brief descriptions`
    prompt += `\n- Accommodation suggestions${budget ? ' within the specified budget' : ''}`
    prompt += `\n- Transportation options and travel tips`
    prompt += `\n- Local cuisine recommendations`
    prompt += `\n- Important travel information and safety tips`
    
    if (!budget) {
      prompt += `\n- Budget estimates for different expense categories`
    }
    
    // Navigate to chat with the generated prompt
    const queryParams = new URLSearchParams()
    queryParams.set('message', prompt)
    
    // Clear any existing fromChat flag when navigating TO chat
    sessionStorage.removeItem('fromChat')
    
    const url = `/chat?${queryParams.toString()}`
    router.push(url)
  }

  const handleInputChange = (field: string, value: string) => {
    // For integer fields, only allow numbers
    if (field === 'days' || field === 'budget' || field === 'travellers') {
      const numericValue = value.replace(/[^0-9]/g, '')
      setPlanningForm(prev => ({...prev, [field]: numericValue}))
    } else {
      setPlanningForm(prev => ({...prev, [field]: value}))
    }
  }

  const handleQuickMessage = () => {
    if (quickMessage.trim()) {
      // Clear any existing fromChat flag when navigating TO chat
      sessionStorage.removeItem('fromChat')
      router.push(`/chat?message=${encodeURIComponent(quickMessage.trim())}`)
    }
  }

  const getUserDisplayName = () => {
    if (user?.user_metadata?.first_name && user?.user_metadata?.last_name) {
      return `${user.user_metadata.first_name} ${user.user_metadata.last_name}`
    }
    return user?.user_metadata?.username || user?.user_metadata?.first_name || user?.email?.split('@')[0] || 'User'
  }

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut()
      router.push("/auth/login")
    } catch (error) {
      console.error('Sign out error:', error)
      router.push("/auth/login")
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-50 via-white to-indigo-50">
        <div className="text-center">
          {/* Roameo Logo Animation */}
          <div className="mb-8 relative">
            <div className="w-20 h-20 bg-black rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
              <div className="w-8 h-8 bg-white rounded-full animate-ping"></div>
            </div>
            <div className="absolute inset-0 w-20 h-20 bg-black rounded-full mx-auto opacity-20 animate-ping"></div>
          </div>
          
          {/* Roameo Text */}
          <h2 className="text-3xl font-bold text-gray-900 mb-2 animate-fade-in">
            roameo
          </h2>
          <p className="text-gray-600 mb-6 animate-fade-in-delay">
            Your Intelligent Travel CoPilot
          </p>
          
          {/* Loading Animation */}
          <div className="flex justify-center items-center space-x-2 mb-4">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
          
          <p className="text-gray-500 text-sm animate-pulse">Loading your personalized travel dashboard...</p>
        </div>
        
        <style jsx>{`
          @keyframes fade-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          .animate-fade-in {
            animation: fade-in 1s ease-out;
          }
          .animate-fade-in-delay {
            animation: fade-in 1s ease-out 0.5s both;
          }
        `}</style>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white px-20 py-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center relative overflow-hidden">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20 animate-sweep"></div>
          </div>
          <span className="text-xl font-bold text-gray-900 tracking-tight">roameo</span>
        </div>
        
        <div className="flex items-center gap-4">
          <span className="font-prompt text-l font-small text-gray-900">
            Welcome, {getUserDisplayName()}
          </span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="border-gray-300 bg-white hover:bg-gray-50 w-8 h-8 p-0 rounded-full"
              >
                <User className="w-4 h-4 text-gray-700" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52 bg-white border border-gray-200 shadow-xl rounded-lg p-2 z-[10001]">
              <DropdownMenuItem asChild>
                <Link href="/profile" className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 rounded-lg p-3 transition-all">
                  <User className="w-4 h-4 text-blue-600" />
                  <span className="font-medium">Profile</span>
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator className="my-2 bg-gray-100" />
              <DropdownMenuItem asChild>
                <Link 
                  href="https://github.com/yk0007/Roameo/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 rounded-lg p-3 transition-all"
                >
                  <Github className="w-4 h-4 text-gray-700" />
                  <span className="font-medium">GitHub</span>
                  <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link 
                  href="https://yk0007.pages.dev/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 rounded-lg p-3 transition-all"
                >
                  <Heart className="w-4 h-4 text-red-500" />
                  <span className="font-medium">Meet me</span>
                  <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator className="my-2 bg-gray-100" />
              <DropdownMenuItem 
                className="flex items-center gap-3 cursor-pointer text-red-600 hover:bg-red-50 rounded-lg p-3 transition-all"
                onClick={handleSignOut}
              >
                <LogOut className="w-4 h-4 text-red-600" />
                <span className="font-medium">Sign Out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Hero Section with Trip Planning */}
      <section className="relative px-20 pt-6">
        {/* Background Image */}
        <div 
          className="relative h-[400px] bg-cover bg-center overflow-hidden"
          style={{
            backgroundImage: `url('/hero-palm-background.jpg')`,
            backgroundPosition: 'center center',
            borderRadius: '48px'
          }}
        >
          {/* Overlay for text visibility */}
          <div className="absolute inset-0 bg-black/40" />
          
          {/* Hero Text - Centered */}
          <div className={`absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none ${
            isFromChat 
              ? 'translate-y-0 opacity-100' // No animation if from chat
              : `transition-all duration-1000 transform ${
                  heroVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
                }`
          }`}>
            <span className="font-roboto-mono text-white text-sm font-medium mb-1 animate-pulse">
              I'm
            </span>
            <h1 className="text-6xl font-extrabold text-white leading-tight drop-shadow-sm mb-2 bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text animate-gradient-x">roameo</h1>
            <div className="font-pacifico text-white text-xl font-normal min-h-[1.5rem] flex items-center justify-center">
              <span>{fullText}</span>
            </div>
          </div>
        </div>

        {/* Trip Planning Form - Dynamic Position */}
        <div className={`absolute left-1/2 transform -translate-x-1/2 ${
          isFromChat 
            ? 'translate-y-0 opacity-100' // No animation if from chat
            : `transition-all duration-1000 ${
                formVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
              }`
        }`} style={{
          top: activeInput === 'form' ? 'calc(400px + 24px - 40px)' : activeInput === 'none' ? 'calc(400px + 24px - 40px)' : 'calc(400px + 80px)',
          width: activeInput === 'form' ? '844px' : activeInput === 'none' ? '844px' : '600px',
          zIndex: activeInput === 'form' ? 20 : activeInput === 'none' ? 10 : 10
        }}>
          <div className="relative">
            {/* Form Pill */}
            <div 
              className="flex items-center gap-6 bg-white backdrop-blur border border-black/5 px-8 py-5 transition-all duration-500 cursor-pointer hover:shadow-2xl hover:scale-[1.02] group" 
              style={{
                borderRadius: '100px',
                height: activeInput === 'form' ? '80px' : activeInput === 'none' ? '80px' : '60px',
                boxShadow: '0px 1px 12px rgba(3,3,3,0.1)',
                transform: activeInput === 'form' ? 'scale(1)' : activeInput === 'none' ? 'scale(1)' : 'scale(0.85)'
              }}
              onClick={() => setActiveInput('form')}
              onFocus={() => setActiveInput('form')}
            >
                {/* Destination */}
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-700 mb-1 group-hover:text-black transition-colors duration-300">Destination</div>
                  <Input 
                    placeholder="Where are you going?"
                    value={planningForm.destination}
                    onChange={(e) => handleInputChange('destination', e.target.value)}
                    onFocus={() => setActiveInput('form')}
                    className="w-full bg-transparent border-0 outline-none placeholder:text-gray-400 text-gray-900 h-6 p-0 focus-visible:ring-0 shadow-none transition-all duration-300 focus:scale-[1.01]"
                    required
                  />
                </div>

                {/* Divider */}
                <div className="h-10 w-px bg-gray-200 group-hover:bg-black transition-colors duration-300" />

                {/* Days */}
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-900 mb-1 group-hover:text-black transition-colors duration-300">Days</div>
                  <Input 
                    placeholder="No. of days"
                    value={planningForm.days}
                    onChange={(e) => handleInputChange('days', e.target.value)}
                    onFocus={() => setActiveInput('form')}
                    className="w-full bg-transparent border-0 outline-none placeholder:text-gray-400 text-gray-900 h-6 p-0 focus-visible:ring-0 shadow-none transition-all duration-300 focus:scale-[1.01]"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    required
                  />
                </div>

                {/* Divider */}
                <div className="h-10 w-px bg-gray-200 group-hover:bg-black transition-colors duration-300" />

                {/* Budget */}
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-900 mb-1 group-hover:text-black transition-colors duration-300">Budget</div>
                  <Input 
                    placeholder="INR (Optional)"
                    value={planningForm.budget}
                    onChange={(e) => handleInputChange('budget', e.target.value)}
                    onFocus={() => setActiveInput('form')}
                    className="w-full bg-transparent border-0 outline-none placeholder:text-gray-400 text-gray-900 h-6 p-0 focus-visible:ring-0 shadow-none transition-all duration-300 focus:scale-[1.01]"
                    inputMode="numeric"
                    pattern="[0-9]*"
                  />
                </div>

                {/* Divider */}
                <div className="h-10 w-px bg-gray-200 group-hover:bg-black transition-colors duration-300" />

                {/* Trip Type */}
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-900 mb-1 group-hover:text-black transition-colors duration-300">Trip Type</div>
                  <Select 
                    value={planningForm.travellers} 
                    onValueChange={(value) => handleInputChange('travellers', value)}
                    onOpenChange={() => setActiveInput('form')}
                  >
                    <SelectTrigger 
                      className="w-full bg-transparent border-0 outline-none h-6 p-0 focus:ring-0 shadow-none transition-all duration-300 hover:scale-[1.01]"
                      onFocus={() => setActiveInput('form')}
                    >
                      <SelectValue placeholder="Nature" className="text-gray-400" />
                    </SelectTrigger>
                    <SelectContent className="bg-white border-0 shadow-lg rounded-lg animate-fade-in">
                      <SelectItem value="adventure" className="hover:bg-blue-50 transition-colors duration-200">Adventure</SelectItem>
                      <SelectItem value="luxury" className="hover:bg-blue-50 transition-colors duration-200">Luxury</SelectItem>
                      <SelectItem value="budget" className="hover:bg-blue-50 transition-colors duration-200">Budget</SelectItem>
                      <SelectItem value="family" className="hover:bg-blue-50 transition-colors duration-200">Family</SelectItem>
                      <SelectItem value="romantic" className="hover:bg-blue-50 transition-colors duration-200">Romantic</SelectItem>
                      <SelectItem value="cultural" className="hover:bg-blue-50 transition-colors duration-200">Cultural</SelectItem>
                      <SelectItem value="nature" className="hover:bg-blue-50 transition-colors duration-200">Nature</SelectItem>
                      <SelectItem value="business" className="hover:bg-blue-50 transition-colors duration-200">Business</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Black Round Send Button */}
                <Button 
                  onClick={handlePlanTrip}
              className="ml-4 h-12 w-12 flex items-center justify-center rounded-full bg-black text-white p-0 shadow-none hover:bg-gray-800 hover:scale-110 transition-all duration-300 group/btn"
            >
              <ArrowRight className="w-5 h-5 text-white group-hover/btn:translate-x-0.5 transition-transform duration-300" />
            </Button>
              </div>


            </div>
          </div>
        
        {/* Caption between form and chat when activeInput is 'none' */}
        {activeInput === 'none' && (
          <div className="absolute left-1/2 transform -translate-x-1/2" style={{
            top: 'calc(400px + 67px)',
            zIndex: 15
          }}>
            <div className="bg-black text-white px-4 py-1 text-sm font-medium text-center" style={{
              borderRadius: '20px',
              fontSize: '12px'
            }}>
              Fill form or Chat with roameo
            </div>
          </div>
        )}

        {/* Chat Input - Dynamic Position */}
        <div className={`absolute left-1/2 transform -translate-x-1/2 ${
          isFromChat 
            ? 'translate-y-0 opacity-100' // No animation if from chat
            : `transition-all duration-1000 ${
                formVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
              }`
        }`} style={{
          top: activeInput === 'chat' ? 'calc(400px + 24px - 40px)' : activeInput === 'none' ? 'calc(400px + 100px)' : 'calc(400px + 80px)',
          width: activeInput === 'chat' ? '844px' : activeInput === 'none' ? '844px' : '600px',
          zIndex: activeInput === 'chat' ? 20 : activeInput === 'none' ? 10 : 10
        }}>
          <div 
            className="flex items-center bg-white border border-black/5 px-8 py-6 transition-all duration-500 cursor-pointer hover:shadow-2xl hover:scale-[1.02] group" 
            style={{
              borderRadius: '100px',
              height: activeInput === 'chat' ? '83px' : activeInput === 'none' ? '83px' : '60px',
              boxShadow: '0px 1px 12px rgba(3,3,3,0.1)',
              transform: activeInput === 'chat' ? 'scale(1)' : activeInput === 'none' ? 'scale(1)' : 'scale(0.85)'
            }}
            onClick={() => setActiveInput('chat')}
            onFocus={() => setActiveInput('chat')}
          >
            <Input 
              placeholder="Where would you like to go?"
              value={quickMessage}
              onChange={(e) => setQuickMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleQuickMessage()}
              onFocus={() => setActiveInput('chat')}
              className="flex-1 bg-transparent border-0 outline-none placeholder:text-gray-400 text-gray-900 h-6 p-0 focus-visible:ring-0 shadow-none transition-all duration-300 focus:scale-[1.01] group-hover:placeholder:text-blue-400"
            />
            <Button 
              onClick={handleQuickMessage}
              className="ml-4 h-12 w-12 flex items-center justify-center rounded-full bg-black text-white p-0 shadow-none hover:bg-gray-800 hover:scale-110 transition-all duration-300 group/btn"
            >
              <ArrowRight className="w-5 h-5 text-white group-hover/btn:translate-x-0.5 transition-transform duration-300" />
            </Button>
          </div>
        </div>
      </section>

      {/* Your Trips Section */}
      <section className={`px-20 py-12 ${
        isFromChat 
          ? 'translate-y-0 opacity-100' // No animation if from chat
          : `transition-all duration-1000 transform ${
              tripsVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
            }`
      }`} style={{ marginTop: 'calc(150px )' }}>
        <h2 className="font-prompt text-xl font-medium text-gray-900 mb-8 bg-black bg-clip-text text-transparent">
          Your Trips
        </h2>
        
        {trips.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {trips.map((trip, index) => {
              const gradients = [
                'from-pink-400 to-red-500',
                'from-blue-400 to-indigo-500', 
                'from-yellow-400 to-orange-500',
                'from-green-400 to-emerald-500',
                'from-purple-400 to-pink-500',
                'from-indigo-400 to-purple-500',
                'from-gray-400 to-gray-600',
                'from-orange-400 to-red-500'
              ]
              const gradientClass = gradients[index % gradients.length]
              
              return (
                <Card 
                  key={trip.id}
                  className={`group hover:shadow-2xl transition-all duration-500 cursor-pointer bg-white rounded-3xl border border-gray-200 overflow-hidden transform hover:scale-[1.03] hover:-translate-y-2 ${
                    isFromChat 
                      ? 'translate-y-0 opacity-100' // No animation if from chat
                      : (tripsVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0')
                  }`}
                  style={isFromChat ? {} : { 
                    animationDelay: `${index * 100}ms`,
                    transitionDelay: `${index * 100}ms`
                  }}
                  onClick={() => {
                    // Clear any existing fromChat flag when navigating TO chat
                    sessionStorage.removeItem('fromChat')
                    router.push(`/chat?sessionId=${encodeURIComponent(trip.id)}`)
                  }}
                >
                  <CardContent className="p-6">
                    <div className="aspect-square rounded-2xl mb-4 overflow-hidden relative group-hover:scale-110 transition-transform duration-500">
                      {/* Image priority: 1. destinationImageUrl (Google Places), 2. Fallback to colorful gradient */}
                      {trip.destinationImageUrl ? (
                        // Use destination image if available
                        <CachedImage 
                          src={trip.destinationImageUrl}
                          alt={trip.destination || trip.title || 'Trip destination'}
                          className="w-full h-full object-cover"
                          onError={() => {
                            console.log(`Failed to load destination image for ${trip.destination}`);
                          }}
                        />
                      ) : (
                        // Fallback to colorful gradient card
                        <div className={`w-full h-full bg-gradient-to-br ${gradientClass} flex items-center justify-center`}>
                          <span className="text-white text-2xl font-bold group-hover:scale-110 transition-transform duration-300">
                            {trip.title?.charAt(0) || trip.destination?.charAt(0) || '?'}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    <h3 className="font-semibold text-gray-900 text-lg mb-2 group-hover:text-blue-600 transition-colors duration-300">{trip.title || 'Untitled Trip'}</h3>
                    <p className="text-gray-600 text-sm mb-1 group-hover:text-gray-800 transition-colors duration-300">Destination: {trip.destination || 'Not specified'}</p>
                    <p className="text-gray-600 text-sm group-hover:text-gray-800 transition-colors duration-300">Days: {trip.days || 'Not specified'}</p>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        ) : (
          <div className="text-center py-12 animate-fade-in">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4 hover:scale-110 transition-transform duration-300 animate-bounce">
              <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No trips yet</h3>
            <p className="text-gray-600 mb-6">Start planning your first adventure using the form above!</p>
            <Button 
              onClick={handlePlanTrip}
              className="bg-black hover:bg-gray-800 text-white px-6 py-2 rounded-full hover:scale-105 transition-all duration-300 group"
            >
              <ArrowRight className="w-4 h-4 mr-2 group-hover:translate-x-1 transition-transform duration-300" />
              Plan Your First Trip
            </Button>
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 border-t border-gray-200 bg-white">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-6 h-6 bg-black rounded-full flex items-center justify-center">
                <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
              </div>
              <span className="text-lg font-bold text-gray-900">roameo</span>
            </div>
            <p className="text-sm text-gray-600">Your Intelligent Travel CoPilot</p>
            <p className="text-xs text-gray-400 mt-2">roameo © 2025</p>
          </div>
          
          <div className="text-right">
            <p className="text-sm text-gray-600">Contact us</p>
          </div>
        </div>
      </footer>
      
      {/* Font imports and custom CSS for animations */}
      <style jsx>{`
        @keyframes sweep {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-sweep {
          animation: sweep 2s ease-in-out infinite;
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }
        .animate-fade-in-delay {
          animation: fade-in 1s ease-out 0.5s both;
        }
      `}</style>
    </div>
  )
}
