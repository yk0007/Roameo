import { ChevronDown, X, ArrowLeft, MoreHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"

interface TripHeaderProps {
  trip: {
    title: string
    location: string
    dates: string
    travelers: string
    budget: string
  }
}

export function TripHeader({ trip }: TripHeaderProps) {
  return (
    <div className="bg-white border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" className="p-1">
            <ArrowLeft className="w-4 h-4" />
          </Button>

          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">{trip.title}</h1>
            <ChevronDown className="w-4 h-4 text-gray-500" />
          </div>

          <div className="flex items-center gap-4 text-sm text-gray-600">
            <span>{trip.location}</span>
            <span>{trip.dates}</span>
            <span>{trip.travelers}</span>
            <span>{trip.budget}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm">
            <X className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="sm">
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="sm">
            <MoreHorizontal className="w-4 h-4" />
          </Button>

          <Button className="bg-blue-600 hover:bg-blue-700 text-white">Invite</Button>

          <Button variant="outline" className="gap-2 bg-transparent">
            <span className="w-4 h-4 bg-blue-600 rounded text-white text-xs flex items-center justify-center">14</span>
            Unplanned Trip
          </Button>
        </div>
      </div>
    </div>
  )
}
