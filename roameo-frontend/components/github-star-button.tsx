"use client"

import { Button } from "@/components/ui/button"
import { MoreHorizontal, Github, Heart, ExternalLink } from "lucide-react"
import Link from "next/link"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface MoreOptionsButtonProps {
  variant?: "default" | "ghost" | "outline"
  size?: "default" | "sm" | "lg"
  className?: string
}

export function MoreOptionsButton({ 
  variant = "outline", 
  size = "sm", 
  className = ""
}: MoreOptionsButtonProps) {
  const githubUrl = "https://github.com/yk0007/Roameo/"
  const meetMeUrl = "https://yk0007.pages.dev/"
  
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant={variant} 
          size={size}
          className={`flex items-center gap-2 hover:bg-gray-100 border-gray-300 text-gray-700 hover:text-gray-900 transition-all duration-200 hover:shadow-md hover:scale-105 ${className}`}
        >
          <MoreHorizontal className="w-4 h-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="center" className="w-48 bg-white border border-gray-200 shadow-xl rounded-lg p-2 z-[10001]">
        <DropdownMenuItem asChild>
          <Link 
            href={githubUrl} 
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
            href={meetMeUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 rounded-lg p-3 transition-all"
          >
            <Heart className="w-4 h-4 text-red-500" />
            <span className="font-medium">Meet me</span>
            <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}