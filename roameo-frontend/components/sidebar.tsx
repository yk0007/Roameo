import { MessageCircle, Search, Heart, MapPin, Bell, Lightbulb, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

export function Sidebar() {
  const trips = [
    {
      id: "1",
      title: "Kanyakumari Travel Tips",
      image: "/kanyakumari-beach.png",
    },
    {
      id: "2",
      title: "Araku Valley: Scenic 2-Day Road Trip...",
      image: "/placeholder-mb1ek.png",
      active: true,
    },
  ]

  const navItems = [
    { icon: MessageCircle, label: "Chats", count: 2 },
    { icon: Search, label: "Explore" },
    { icon: Heart, label: "Saved" },
    { icon: MapPin, label: "Trips" },
    { icon: Bell, label: "Updates" },
    { icon: Lightbulb, label: "Inspiration" },
    { icon: Plus, label: "Create" },
  ]

  return (
    <div className="w-64 bg-gray-50 border-r border-border flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-6 h-6 bg-black rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">⚡</span>
          </div>
          <span className="font-semibold text-lg">mindtrip.</span>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex-1 p-4">
        <nav className="space-y-2">
          {navItems.map((item) => (
            <Button
              key={item.label}
              variant={item.label === "Chats" ? "default" : "ghost"}
              className={`w-full justify-start gap-3 ${
                item.label === "Chats"
                  ? "bg-black text-white hover:bg-gray-800"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
              }`}
            >
              <item.icon className="w-4 h-4" />
              <span>{item.label}</span>
              {item.count && (
                <span className="ml-auto bg-gray-600 text-white text-xs rounded-full px-2 py-0.5">{item.count}</span>
              )}
            </Button>
          ))}
        </nav>

        {/* Trips Section */}
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-500">All</h3>
            <span className="text-sm text-gray-400">Trips</span>
          </div>

          <div className="space-y-2">
            {trips.map((trip) => (
              <div
                key={trip.id}
                className={`flex items-center gap-3 p-2 rounded-lg cursor-pointer ${
                  trip.active ? "bg-white shadow-sm" : "hover:bg-gray-100"
                }`}
              >
                <img src={trip.image || "/placeholder.svg"} alt="" className="w-10 h-10 rounded-lg object-cover" />
                <span className="text-sm font-medium text-gray-900 truncate">{trip.title}</span>
              </div>
            ))}
          </div>
        </div>

        {/* New Chat Button */}
        <Button
          variant="outline"
          className="w-full mt-6 border-dashed border-gray-300 text-gray-600 hover:text-gray-900 bg-transparent"
        >
          New chat
        </Button>
      </div>

      {/* User Profile */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-3">
          <Avatar className="w-8 h-8">
            <AvatarFallback className="bg-orange-500 text-white text-sm">N</AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900">Neil</p>
            <p className="text-xs text-gray-500 truncate">@neil.d2373</p>
          </div>
        </div>
      </div>
    </div>
  )
}
