import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Clock, MapPin, Star, Camera } from "lucide-react"
import { CachedImage } from "./cached-image"
import { resolvePoiImageUrl } from "@/lib/poi-image-url"
import type { ItineraryDay } from "@/lib/types"

interface DayHoverCardProps {
  day: ItineraryDay
  onClose?: () => void
  onDetailClick?: (activity: any) => void
}

export function DayHoverCard({ day, onClose, onDetailClick }: DayHoverCardProps) {
  if (!day || typeof day.day !== 'number') return null

  return (
    <Card className="w-80 max-h-96 overflow-y-auto shadow-lg border-2">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Day {day.day}</CardTitle>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose} className="h-6 w-6 p-0">
              ×
            </Button>
          )}
        </div>
        {day.title && <p className="text-sm text-gray-600">{day.title}</p>}
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Activities */}
        {day.activities?.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-medium text-sm flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Activities ({day.activities.length})
            </h4>
            
            {day.activities.map((activity, index) => {
              if (!activity || !activity.name) return null
              
              return (
                <div key={index} className="flex gap-3 p-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors">
                  <div className="w-10 h-10 rounded-lg overflow-hidden bg-gray-200 flex items-center justify-center flex-shrink-0">
                    {resolvePoiImageUrl(activity.photoUrl) ? (
                      <CachedImage
                        src={resolvePoiImageUrl(activity.photoUrl)}
                        alt={activity.name}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <Camera className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h5 className="font-medium text-sm truncate">{activity.name}</h5>
                        
                        {activity.start && activity.end && (
                          <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
                            <Clock className="w-3 h-3" />
                            {activity.start} - {activity.end}
                          </div>
                        )}
                        
                        {activity.location && (
                          <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
                            <MapPin className="w-3 h-3" />
                            <span className="truncate">{activity.location}</span>
                          </div>
                        )}
                        
                        {activity.rating && (
                          <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
                            <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                            {activity.rating}
                          </div>
                        )}
                      </div>
                      
                      {onDetailClick && (
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="text-xs h-6 px-2 ml-2"
                          onClick={() => onDetailClick(activity)}
                        >
                          Detail
                        </Button>
                      )}
                    </div>
                    
                    {activity.description && (
                      <p className="text-xs text-gray-600 mt-1 line-clamp-2">{activity.description}</p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
        
        {/* Accommodation */}
        {day.accommodation && day.accommodation.name && (
          <div className="space-y-2">
            <h4 className="font-medium text-sm flex items-center gap-2">
              🏨 Accommodation
            </h4>
            <div className="flex items-center gap-3 p-2 bg-blue-50 rounded-lg">
              <div className="w-8 h-8 bg-blue-200 rounded flex items-center justify-center">
                <span className="text-sm">🏨</span>
              </div>
              <div className="flex-1">
                <h5 className="font-medium text-sm">{day.accommodation.name}</h5>
                {day.accommodation.checkIn && (
                  <p className="text-xs text-gray-500">Check-in: {day.accommodation.checkIn}</p>
                )}
                {day.accommodation.location && (
                  <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
                    <MapPin className="w-3 h-3" />
                    {day.accommodation.location}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        {/* Summary/Theme */}
        {day.theme && (
          <div className="pt-2 border-t">
            <Badge variant="secondary" className="text-xs">
              {day.theme}
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
