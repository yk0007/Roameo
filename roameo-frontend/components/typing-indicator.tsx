"use client"

import { TextShimmer } from '@/components/ui/text-shimmer'

interface TypingIndicatorProps {
  isVisible: boolean
}

export function TypingIndicator({ isVisible }: TypingIndicatorProps) {
  if (!isVisible) return null

  return (
    <div className="flex items-center gap-2 text-sm text-gray-600">
      <div className="flex space-x-1">
        <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
        <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
        <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div>
      </div>
      <TextShimmer 
        className="text-sm font-medium [--base-color:theme(colors.gray.600)] [--base-gradient-color:theme(colors.gray.400)]" 
        duration={1.5}
      >
        thinking...
      </TextShimmer>
    </div>
  )
}
