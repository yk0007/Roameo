"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import Link from "next/link"
import type { AuthChangeEvent, Session } from "@supabase/supabase-js"
import { redirectToLogin } from "@/lib/auth-redirect"
import { supabase } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { listTrips } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { DashboardTripCard, type DashboardTripSummary } from "@/components/dashboard-trip-card"
import { ArrowRight, User, LogOut, Github, Heart, ExternalLink } from "lucide-react"
import { DotDistortionShaderBg } from "@/components/ui/dot-distortion-shader-bg"

export default function Dashboard() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [trips, setTrips] = useState<DashboardTripSummary[]>([])
  const [planningForm, setPlanningForm] = useState({
    destination: '',
    days: '',
    budget: '',
    travellers: 'nature' // Set Nature as default
  })
  const [quickMessage, setQuickMessage] = useState('')
  const [activeInput, setActiveInput] = useState('none') // 'form', 'chat', or 'none'
  const [activeSection, setActiveSection] = useState<'form' | 'chat'>('form')
  const [validationDialog, setValidationDialog] = useState<{ title: string; message: string } | null>(null)
  
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
        redirectToLogin()
        return
      }
      setUser(session.user)
      setLoading(false)
    }

    getUser()

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session: Session | null) => {
      if (!session) {
        redirectToLogin()
      } else {
        setUser(session.user)
      }
    })

    return () => subscription.unsubscribe()
  }, [router])

  const handlePlanTrip = () => {
    // Validate required fields
    if (!planningForm.destination.trim()) {
      setValidationDialog({
        title: "Destination needed",
        message: "Enter a destination before generating a trip plan.",
      })
      return
    }
    if (!planningForm.days.trim()) {
      setValidationDialog({
        title: "Trip duration needed",
        message: "Add the number of days so Roameo can pace the itinerary correctly.",
      })
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
    if (field === 'days' || field === 'budget') {
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
      <div className="flex min-h-screen items-center justify-center bg-[#f2f3f4] px-6">
        <div className="flex items-center gap-4 rounded-full bg-white px-6 py-4 shadow-[0_18px_42px_rgba(15,23,42,0.08)]">
          <div className="relative flex h-10 w-10 items-center justify-center overflow-hidden rounded-full bg-black">
            <div className="h-2.5 w-2.5 rounded-full bg-white"></div>
          </div>
          <div>
            <div className="text-[1.2rem] font-semibold tracking-[-0.05em] text-gray-950">roameo</div>
            <div className="mt-0.5 text-xs text-gray-500">Loading your dashboard</div>
          </div>
          <div className="ml-2 flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-black animate-pulse"></div>
            <div className="h-2 w-2 rounded-full bg-black/55 animate-pulse [animation-delay:120ms]"></div>
            <div className="h-2 w-2 rounded-full bg-black/25 animate-pulse [animation-delay:240ms]"></div>
          </div>
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
    );
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-white">
      {/* Radial Gradient Background from Bottom */}
      <div
        className="absolute inset-0 z-0"
        style={{
          background: "radial-gradient(125% 125% at 50% 90%, #fff 40%, #475569 100%)",
        }}
      />
      <div 
        className="absolute inset-0 z-[1]"
        style={{
          WebkitMaskImage: "radial-gradient(ellipse 100% 80% at 50% 100%, transparent 50%, #000 90%)",
          maskImage: "radial-gradient(ellipse 100% 80% at 50% 100%, transparent 50%, #000 90%)",
        }}
      >
        <DotDistortionShaderBg />
      </div>
      {/* Diagonal Cross Grid Bottom Background Overlay */}
      <div
        className="absolute inset-0 z-[2]"
        style={{
          backgroundImage: `
            linear-gradient(45deg, transparent 49%, #e5e7eb 49%, #e5e7eb 51%, transparent 51%),
            linear-gradient(-45deg, transparent 49%, #e5e7eb 49%, #e5e7eb 51%, transparent 51%)
          `,
          backgroundSize: "40px 40px",
          WebkitMaskImage: "radial-gradient(ellipse 100% 80% at 50% 100%, #000 50%, transparent 90%)",
          maskImage: "radial-gradient(ellipse 100% 80% at 50% 100%, #000 50%, transparent 90%)",
          opacity: 0.6,
        }}
      />
      <div className="relative z-10">
      <AnimatePresence>
        {validationDialog ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[12000] flex items-center justify-center bg-[rgba(242,243,244,0.48)] px-6 backdrop-blur-xl"
          >
            <motion.div
              initial={{ opacity: 0, y: 18, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 12, scale: 0.98 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              className="w-full max-w-md rounded-[28px] border border-white/80 bg-white/92 p-7 shadow-[0_28px_90px_rgba(15,23,42,0.16)]"
            >
              <div className="mb-5 flex items-start gap-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-black text-sm font-semibold text-white">
                  !
                </div>
                <div className="flex-1">
                  <h3 className="text-[1.35rem] font-semibold tracking-[-0.04em] text-gray-950">
                    {validationDialog.title}
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-gray-600">
                    {validationDialog.message}
                  </p>
                </div>
              </div>

              <div className="flex justify-end">
                <Button
                  type="button"
                  onClick={() => setValidationDialog(null)}
                  className="h-11 rounded-full bg-black px-5 text-sm font-medium text-white hover:bg-gray-800"
                >
                  Continue
                </Button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <header className="px-4 pb-8 pt-8 sm:px-6 lg:px-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex h-8 w-8 items-center justify-center overflow-hidden rounded-full bg-black">
              <div className="h-2 w-2 rounded-full bg-white"></div>
            </div>
            <span className="text-[1.75rem] font-semibold tracking-[-0.05em] text-gray-950">roameo</span>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-xs uppercase tracking-[0.24em] text-black font-semibold">Workspace</p>
              <p className="mt-1 text-sm font-medium text-gray-900">Welcome, {getUserDisplayName()}</p>
            </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-9 w-9 rounded-full border-gray-200 bg-white p-0 hover:bg-gray-50"
              >
                <User className="w-4 h-4 text-gray-700" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="z-[10001] w-52 rounded-2xl border border-gray-200 bg-white p-2 shadow-xl">
              <DropdownMenuItem asChild>
                <Link href="/profile" className="flex items-center gap-3 cursor-pointer rounded-xl p-3 transition-all hover:bg-gray-50">
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
                  className="flex items-center gap-3 cursor-pointer rounded-xl p-3 transition-all hover:bg-gray-50"
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
                  className="flex items-center gap-3 cursor-pointer rounded-xl p-3 transition-all hover:bg-gray-50"
                >
                  <Heart className="w-4 h-4 text-red-500" />
                  <span className="font-medium">Meet me</span>
                  <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator className="my-2 bg-gray-100" />
              <DropdownMenuItem 
                className="flex items-center gap-3 cursor-pointer rounded-xl p-3 text-red-600 transition-all hover:bg-red-50"
                onClick={handleSignOut}
              >
                <LogOut className="w-4 h-4 text-red-600" />
                <span className="font-medium">Sign Out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        </div>
      </header>

      <section className="relative px-4 pb-[230px] pt-2 sm:px-6 lg:px-8">
        <div 
          className="relative mx-auto h-[400px] max-w-6xl overflow-hidden border-[10px] border-white bg-cover bg-center shadow-[0_28px_80px_rgba(15,23,42,0.16)]"
          style={{
            backgroundImage: `url('/tropical-glass.jpeg')`,
            backgroundPosition: 'center 56%',
            borderRadius: '40px'
          }}
        >
          <div className="absolute inset-0 bg-black/40" />

          <div className={`absolute inset-0 z-10 flex flex-col items-center justify-center text-center pointer-events-none ${
            isFromChat 
              ? 'translate-y-0 opacity-100'
              : `transition-all duration-1000 transform ${
                  heroVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
                }`
          }`}>
            <span className="font-roboto-mono mb-1 text-sm font-medium text-white animate-pulse">
              I'm
            </span>
            <h1 className="text-6xl font-extrabold text-white leading-tight drop-shadow-sm mb-2 bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text animate-gradient-x">roameo</h1>
            <div className="mt-4 min-h-[1.5rem] flex items-center justify-center">
              <span className="font-pacifico max-w-xl text-xl font-normal text-white">{fullText}</span>
            </div>
          </div>
        </div>

        {/* Trip Planning Form - Dynamic Position */}
        <motion.div 
          layout
          className={`absolute left-0 right-0 mx-auto ${
            isFromChat
              ? 'translate-y-0 opacity-100'
              : (formVisible ? 'transition-all duration-1000 translate-y-0 opacity-100' : 'transition-all duration-1000 translate-y-20 opacity-0')
          }`} 
          animate={{
            scale: activeSection === 'form' ? 1 : 0.95,
            y: activeSection === 'form' ? 0 : 20,
          }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          style={{
            top: activeInput === 'form' ? 'calc(400px + 18px - 40px)' : activeInput === 'none' ? 'calc(400px + 18px - 40px)' : 'calc(400px + 94px)',
            width: activeInput === 'form' ? '844px' : activeInput === 'none' ? '844px' : '600px',
            zIndex: activeSection === 'form' ? 20 : (activeInput === 'form' ? 20 : activeInput === 'none' ? 10 : 10)
          }}
          onClick={() => setActiveSection('form')}
        >
          <div className="relative">
            {/* Form Pill */}
            <div 
              className="group relative flex cursor-pointer items-center gap-6 overflow-hidden border border-black/5 bg-[#ffffff] px-8 py-5 transition-all duration-500 hover:scale-[1.01] hover:shadow-[0_28px_60px_rgba(15,23,42,0.14)]" 
              style={{
                borderRadius: '100px',
                height: activeInput === 'form' ? '80px' : activeInput === 'none' ? '80px' : '60px',
                boxShadow: activeInput === 'form' ? '0px 18px 40px rgba(15,23,42,0.12)' : '0px 10px 28px rgba(15,23,42,0.08)',
                transform: activeInput === 'form' ? 'scale(1)' : activeInput === 'none' ? 'scale(1)' : 'scale(0.85)'
              }}
              onClick={() => setActiveInput('form')}
              onFocus={() => setActiveInput('form')}
            >
              <div className="pointer-events-none absolute inset-y-3 left-[-18%] w-24 rounded-full bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.92),transparent)] opacity-0 blur-xl transition-all duration-700 group-hover:left-[92%] group-hover:opacity-100" />
                {/* Destination */}
                <div className="flex-1 min-w-0">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500 whitespace-nowrap truncate transition-colors duration-300 group-hover:text-gray-700">Destination</div>
                  <Input 
                    placeholder="Where are you going?"
                    value={planningForm.destination}
                    onChange={(e) => handleInputChange('destination', e.target.value)}
                    onFocus={() => setActiveInput('form')}
                    className="h-9 w-full border-0 bg-transparent px-0 text-gray-900 shadow-none outline-none truncate placeholder:truncate placeholder:text-gray-400 focus-visible:ring-0"
                    required
                  />
                </div>

                {/* Divider */}
                <div className="h-10 w-px bg-gray-200 group-hover:bg-black transition-colors duration-300" />

                {/* Days */}
                <div className="flex-1 min-w-0">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500 whitespace-nowrap truncate transition-colors duration-300 group-hover:text-gray-700">Days</div>
                  <Input 
                    placeholder="No. of days"
                    value={planningForm.days}
                    onChange={(e) => handleInputChange('days', e.target.value)}
                    onFocus={() => setActiveInput('form')}
                    className="h-9 w-full border-0 bg-transparent px-0 text-gray-900 shadow-none outline-none truncate placeholder:truncate placeholder:text-gray-400 focus-visible:ring-0"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    required
                  />
                </div>

                {/* Divider */}
                <div className="h-10 w-px bg-gray-200 group-hover:bg-black transition-colors duration-300" />

                {/* Budget */}
                <div className="flex-1 min-w-0">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500 whitespace-nowrap truncate transition-colors duration-300 group-hover:text-gray-700">Budget</div>
                  <Input 
                    placeholder="INR (Optional)"
                    value={planningForm.budget}
                    onChange={(e) => handleInputChange('budget', e.target.value)}
                    onFocus={() => setActiveInput('form')}
                    className="h-9 w-full border-0 bg-transparent px-0 text-gray-900 shadow-none outline-none truncate placeholder:truncate placeholder:text-gray-400 focus-visible:ring-0"
                    inputMode="numeric"
                    pattern="[0-9]*"
                  />
                </div>

                {/* Divider */}
                <div className="h-10 w-px bg-gray-200 group-hover:bg-black transition-colors duration-300" />

                {/* Trip Type */}
                <div className="flex-1 min-w-0">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500 whitespace-nowrap truncate transition-colors duration-300 group-hover:text-gray-700">Trip Type</div>
                  <Select 
                    value={planningForm.travellers} 
                    onValueChange={(value) => handleInputChange('travellers', value)}
                    onOpenChange={() => setActiveInput('form')}
                  >
                    <SelectTrigger 
                      className="h-9 w-full border-0 bg-transparent px-0 focus:ring-0 shadow-none truncate"
                      onFocus={() => setActiveInput('form')}
                    >
                      <SelectValue placeholder="Nature" className="text-gray-400 truncate" />
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
      </motion.div>

      {/* Caption between form and chat when activeInput is 'none' */}
      {activeInput === 'none' && (
          <div className="absolute left-1/2 transform -translate-x-1/2" style={{
            top: 'calc(400px + 104px)',
            zIndex: 15
          }}>
            <div className="rounded-full bg-black px-4 py-1.5 text-center text-[12px] font-medium text-white shadow-[0_10px_18px_rgba(0,0,0,0.16)]">
              Choose a route or start in chat
            </div>
          </div>
        )}

        {/* Chat Input - Dynamic Position */}
        <motion.div 
          layout
          className={`absolute left-0 right-0 mx-auto ${
          isFromChat 
            ? 'translate-y-0 opacity-100' // No animation if from chat
            : `transition-all duration-1000 ${
                formVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
              }`
        }`} 
          animate={{
            scale: activeSection === 'chat' ? 1 : 0.95,
            y: activeSection === 'chat' ? 0 : 20,
          }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          style={{
            top: activeInput === 'chat' ? 'calc(400px + 18px - 40px)' : activeInput === 'none' ? 'calc(400px + 146px)' : 'calc(400px + 94px)',
            width: activeInput === 'chat' ? '844px' : activeInput === 'none' ? '844px' : '600px',
            zIndex: activeSection === 'chat' ? 20 : (activeInput === 'chat' ? 20 : activeInput === 'none' ? 10 : 10)
          }}
          onClick={() => setActiveSection('chat')}
        >
          <div 
            className="group relative flex cursor-pointer items-center overflow-hidden border border-black/5 bg-[#ffffff] px-8 py-6 transition-all duration-500 hover:scale-[1.01] hover:shadow-[0_28px_60px_rgba(15,23,42,0.14)]" 
            style={{
              borderRadius: '100px',
              height: activeInput === 'chat' ? '83px' : activeInput === 'none' ? '83px' : '60px',
              boxShadow: activeInput === 'chat' ? '0px 18px 40px rgba(15,23,42,0.12)' : '0px 10px 28px rgba(15,23,42,0.08)',
              transform: activeInput === 'chat' ? 'scale(1)' : activeInput === 'none' ? 'scale(1)' : 'scale(0.85)'
            }}
            onClick={() => setActiveInput('chat')}
            onFocus={() => setActiveInput('chat')}
          >
            <div className="pointer-events-none absolute inset-y-3 left-[-18%] w-24 rounded-full bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.92),transparent)] opacity-0 blur-xl transition-all duration-700 group-hover:left-[92%] group-hover:opacity-100" />
            <Input 
              placeholder="Where would you like to go?"
              value={quickMessage}
              onChange={(e) => setQuickMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleQuickMessage()}
              onFocus={() => setActiveInput('chat')}
              className="h-10 flex-1 border-0 bg-transparent px-2 text-gray-900 shadow-none outline-none placeholder:text-gray-400 focus-visible:ring-0"
            />
            <Button 
              onClick={handleQuickMessage}
              className="ml-4 h-12 w-12 flex items-center justify-center rounded-full bg-black text-white p-0 shadow-none hover:bg-gray-800 hover:scale-110 transition-all duration-300 group/btn"
            >
              <ArrowRight className="w-5 h-5 text-white group-hover/btn:translate-x-0.5 transition-transform duration-300" />
            </Button>
          </div>
        </motion.div>
      </section>

      {/* Your Trips Section */}
      <section className={`px-4 pb-16 pt-8 sm:px-6 lg:px-8 ${
        isFromChat 
          ? 'translate-y-0 opacity-100'
          : `transition-all duration-1000 transform ${
              tripsVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
            }`
      }`}>
        <div className="mx-auto max-w-6xl">
        <h2 className="mb-10 text-2xl font-semibold tracking-[-0.04em] text-gray-900">
          Your Trips
        </h2>
        
        {trips.length > 0 ? (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
            {trips.map((trip, index) => (
              <DashboardTripCard
                key={trip.id}
                trip={trip}
                index={index}
                animateIn={isFromChat || tripsVisible}
                skipEntryAnimation={isFromChat}
                onClick={() => {
                  sessionStorage.removeItem('fromChat')
                  router.push(`/chat?sessionId=${encodeURIComponent(trip.id)}`)
                }}
              />
            ))}
          </div>
        ) : (
          <div className="py-12 text-center animate-fade-in">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gray-100 transition-transform duration-300">
              <div className="h-8 w-8 rounded-full bg-gray-300/80 animate-pulse"></div>
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
        </div>
      </section>

      {/* Footer */}
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
    </div>
  );
}
