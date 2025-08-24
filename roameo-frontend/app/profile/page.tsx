"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { ArrowLeft, User, Mail, Calendar, MapPin } from "lucide-react"

export default function Profile() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(false)
  const [tripCount, setTripCount] = useState(0)
  const [formData, setFormData] = useState({
    first_name: "",
    last_name: "",
    username: "",
  })

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
      // Handle Google OAuth user metadata structure
      const metadata = session.user.user_metadata || {}
      const fullName = metadata.full_name || metadata.name || ""
      const [firstName, ...lastNameParts] = fullName.split(" ")
      
      setFormData({
        first_name: metadata.first_name || firstName || "",
        last_name: metadata.last_name || lastNameParts.join(" ") || "",
        username: metadata.username || metadata.preferred_username || "",
      })
      
      // Fetch user statistics
      try {
        const response = await fetch('/api/user/stats', {
          headers: {
            'Authorization': `Bearer ${session.access_token}`
          }
        })
        if (response.ok) {
          const stats = await response.json()
          setTripCount(stats.tripCount)
        }
      } catch (error) {
        console.error('Failed to fetch user stats:', error)
      }
      
      setLoading(false)
    }

    getUser()
  }, [router])

  const handleUpdateProfile = async () => {
    try {
      const { error } = await supabase.auth.updateUser({
        data: formData,
      })

      if (error) throw error

      setEditing(false)
      // Refresh user data
      const {
        data: { session },
      } = await supabase.auth.getSession()
      setUser(session?.user)
    } catch (error) {
      console.error("Error updating profile:", error)
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your profile...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen w-full bg-[#f8fafc] relative">
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

      <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-0 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center shadow-md">
                <div className="w-2 h-2 bg-white rounded-full"></div>
              </div>
              <span className="text-xl font-bold text-gray-900">roameo</span>
            </div>
          </div>
        </div>
      </header>

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pt-24">
        <div className="mb-6">
          <Button
            onClick={() => router.push("/dashboard")}
            variant="ghost"
            size="sm"
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 bg-white/60 backdrop-blur-sm border-0 rounded-full shadow-md hover:shadow-lg transition-shadow"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>

        <div className="mb-8 opacity-0 animate-fade-in-up">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Profile Settings</h1>
          <p className="text-gray-600">Manage your account information and preferences</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 opacity-0 animate-fade-in-up animation-delay-200">
            <Card className="bg-white/60 backdrop-blur-sm border-0 shadow-xl rounded-3xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-gray-900">
                  <User className="h-5 w-5" />
                  Personal Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6 rounded-3xl border-0">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="first_name" className="text-gray-700">
                      First Name
                    </Label>
                    <Input
                      id="first_name"
                      value={formData.first_name}
                      onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                      disabled={!editing}
                      className="mt-1 bg-white/80 border-gray-200"
                    />
                  </div>
                  <div>
                    <Label htmlFor="last_name" className="text-gray-700">
                      Last Name
                    </Label>
                    <Input
                      id="last_name"
                      value={formData.last_name}
                      onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                      disabled={!editing}
                      className="mt-1 bg-white/80 border-gray-200"
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="username" className="text-gray-700">
                    Username
                  </Label>
                  <Input
                    id="username"
                    value={formData.username}
                    onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                    disabled={!editing}
                    className="mt-1 bg-white/80 border-gray-200"
                  />
                </div>

                <div>
                  <Label htmlFor="email" className="text-gray-700">
                    Email
                  </Label>
                  <Input
                    id="email"
                    value={user?.email || ""}
                    disabled
                    className="mt-1 bg-gray-100/80 border-gray-200"
                  />
                  <p className="text-xs text-gray-500 mt-1">Email cannot be changed</p>
                </div>

                <div className="flex gap-3">
                  {editing ? (
                    <>
                      <Button onClick={handleUpdateProfile} className="bg-gray-900 hover:bg-gray-800 text-white shadow-md hover:shadow-lg transition-shadow rounded-xl">
                        Save Changes
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => setEditing(false)}
                        className="border-0 text-gray-700 hover:bg-gray-50 shadow-md hover:shadow-lg transition-shadow rounded-xl"
                      >
                        Cancel
                      </Button>
                    </>
                  ) : (
                    <Button onClick={() => setEditing(true)} className="bg-gray-900 hover:bg-gray-800 text-white shadow-md hover:shadow-lg transition-shadow rounded-xl">
                      Edit Profile
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="mt-6 bg-white/60 backdrop-blur-sm border-0 shadow-xl rounded-3xl">
              <CardHeader>
                <CardTitle className="text-gray-900">Security</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="current_password" className="text-gray-700">
                      Current Password
                    </Label>
                    <Input
                      id="current_password"
                      type="password"
                      placeholder="Enter current password"
                      className="mt-1 bg-white/80 border-gray-200"
                    />
                  </div>
                  <div>
                    <Label htmlFor="new_password" className="text-gray-700">
                      New Password
                    </Label>
                    <Input
                      id="new_password"
                      type="password"
                      placeholder="Enter new password"
                      className="mt-1 bg-white/80 border-gray-200"
                    />
                  </div>
                  <div>
                    <Label htmlFor="confirm_password" className="text-gray-700">
                      Confirm New Password
                    </Label>
                    <Input
                      id="confirm_password"
                      type="password"
                      placeholder="Confirm new password"
                      className="mt-1 bg-white/80 border-gray-200"
                    />
                  </div>
                  <Button className="w-full bg-gray-900 hover:bg-gray-800 text-white shadow-md hover:shadow-lg transition-shadow rounded-xl">Change Password</Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Account Summary */}
          <div className="space-y-6 opacity-0 animate-fade-in-up animation-delay-400">
            <Card className="bg-white/60 backdrop-blur-sm border-0 shadow-xl rounded-3xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-gray-900">
                  <Mail className="h-5 w-5" />
                  Account Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-3">
                  <Calendar className="h-4 w-4 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">Member since</p>
                    <p className="text-xs text-gray-600">
                      {user?.created_at
                        ? new Date(user.created_at).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          })
                        : user?.user_metadata?.created_at
                        ? new Date(user.user_metadata.created_at).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          })
                        : user?.user_metadata?.email_verified_at
                        ? new Date(user.user_metadata.email_verified_at).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          })
                        : user?.confirmed_at
                        ? new Date(user.confirmed_at).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          })
                        : "Not available"}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <MapPin className="h-4 w-4 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">Trips planned</p>
                    <p className="text-xs text-gray-600">{tripCount} itineraries</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/60 backdrop-blur-sm border-0 shadow-xl rounded-3xl">
              <CardHeader>
                <CardTitle className="text-gray-900">Travel Preferences</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">Set your travel preferences to get better recommendations</p>
                <Button
                  variant="outline"
                  className="w-full border-0 text-gray-700 hover:bg-white/60 bg-neutral-200 shadow-md hover:shadow-lg transition-shadow rounded-xl"
                >
                  Update Preferences
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
