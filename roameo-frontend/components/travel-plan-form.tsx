"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Loader2, Sparkles } from "lucide-react"
import { BUDGET_OPTIONS } from "@/lib/budget-options"

interface TravelPlanFormProps {
  onPlanGenerated: (itinerary: any) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export function TravelPlanForm({ onPlanGenerated, isLoading, setIsLoading }: TravelPlanFormProps) {
  const [formData, setFormData] = useState({
    destination: "",
    origin: "",
    duration: "",
    budget: "",
    preferences: "",
    travelStyle: "",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    // Simulate API call to Roameo backend
    setTimeout(() => {
      const mockItinerary = {
        destination: formData.destination,
        duration: Number.parseInt(formData.duration),
        dailyPlans: [
          {
            day: 1,
            date: "2024-03-15",
            activities: [
              {
                time: "09:00",
                name: "Araku Valley View Point",
                description: "Breathtaking views of the Araku Valley",
                location: "Araku Valley, Andhra Pradesh",
                duration: 90,
                type: "attraction",
              },
              {
                time: "11:00",
                name: "Borra Caves",
                description: "Million-year-old limestone caves",
                location: "Borra, Andhra Pradesh",
                duration: 120,
                type: "attraction",
              },
            ],
          },
        ],
        totalBudget: formData.budget,
        preferences: formData.preferences.split(",").map((p) => p.trim()),
      }

      onPlanGenerated(mockItinerary)
      setIsLoading(false)
    }, 3000)
  }

  return (
    <div className="p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            AI Travel Planner
          </CardTitle>
          <CardDescription>
            Tell us about your dream trip and let our AI agents create the perfect itinerary
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="destination">Destination</Label>
                <Input
                  id="destination"
                  placeholder="e.g., Araku Valley"
                  value={formData.destination}
                  onChange={(e) => setFormData((prev) => ({ ...prev, destination: e.target.value }))}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="origin">From</Label>
                <Input
                  id="origin"
                  placeholder="e.g., Vizag"
                  value={formData.origin}
                  onChange={(e) => setFormData((prev) => ({ ...prev, origin: e.target.value }))}
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="duration">Duration (days)</Label>
                <Input
                  id="duration"
                  type="number"
                  min="1"
                  max="30"
                  placeholder="3"
                  value={formData.duration}
                  onChange={(e) => setFormData((prev) => ({ ...prev, duration: e.target.value }))}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="budget">Budget</Label>
                <Select
                  value={formData.budget}
                  onValueChange={(value) => setFormData((prev) => ({ ...prev, budget: value }))}
                >
                  <SelectTrigger id="budget">
                    <SelectValue placeholder="Select a budget style" />
                  </SelectTrigger>
                  <SelectContent>
                    {BUDGET_OPTIONS.map((option) => (
                      <SelectItem key={option.id} value={option.label}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="travelStyle">Travel Style</Label>
              <Select
                value={formData.travelStyle}
                onValueChange={(value) => setFormData((prev) => ({ ...prev, travelStyle: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select your travel style" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="adventure">Adventure</SelectItem>
                  <SelectItem value="luxury">Luxury</SelectItem>
                  <SelectItem value="budget">Budget</SelectItem>
                  <SelectItem value="family">Family</SelectItem>
                  <SelectItem value="cultural">Cultural</SelectItem>
                  <SelectItem value="nature">Nature</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="preferences">Preferences & Interests</Label>
              <Textarea
                id="preferences"
                placeholder="e.g., I love nature, local culture, walking, avoid crowded places..."
                value={formData.preferences}
                onChange={(e) => setFormData((prev) => ({ ...prev, preferences: e.target.value }))}
                rows={3}
              />
            </div>

            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  AI Agents Working...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Generate Itinerary
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
