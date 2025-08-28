"use client"

import { memo } from "react"

// Base skeleton component
export const Skeleton = memo(function Skeleton({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`animate-pulse rounded-md bg-gray-200 ${className}`}
      {...props}
    />
  )
})

// POI Card Skeleton
export const PoiCardSkeleton = memo(function PoiCardSkeleton() {
  return (
    <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100">
      <Skeleton className="w-full h-40" />
      <div className="p-3">
        <div className="flex justify-between items-start mb-2">
          <Skeleton className="h-5 w-3/4" />
          <Skeleton className="h-4 w-12" />
        </div>
        <Skeleton className="h-3 w-1/2 mb-2" />
        <Skeleton className="h-3 w-full" />
      </div>
    </div>
  )
})

// Itinerary Day Skeleton
export const ItineraryDaySkeleton = memo(function ItineraryDaySkeleton() {
  return (
    <div className="relative">
      <div className="flex items-center gap-3 p-3 bg-white/95 backdrop-blur-md border-b border-gray-100 shadow-sm">
        <Skeleton className="w-8 h-8 rounded-full" />
        <Skeleton className="h-5 w-32" />
      </div>
      <div className="space-y-3 pl-4 border-l-2 border-zinc-200 ml-4 p-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="relative">
            <div className="absolute left-[-26px] top-5 w-3 h-3 bg-zinc-300 rounded-full border-4 border-white z-10"></div>
            <div className="mb-3 flex items-center gap-2">
              <Skeleton className="w-4 h-4" />
              <Skeleton className="h-4 w-24" />
            </div>
            <PoiCardSkeleton />
          </div>
        ))}
      </div>
    </div>
  )
})

// Chat Message Skeleton
export const ChatMessageSkeleton = memo(function ChatMessageSkeleton({
  isUser = false
}: {
  isUser?: boolean
}) {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${
        isUser 
          ? 'bg-blue-100 text-gray-800 rounded-2xl rounded-br-md border border-blue-200' 
          : 'bg-gray-100 rounded-2xl rounded-bl-md'
      } p-4`}>
        <Skeleton className={`h-4 w-full mb-2 ${isUser ? 'bg-blue-200' : 'bg-gray-300'}`} />
        <Skeleton className={`h-4 w-3/4 ${isUser ? 'bg-blue-200' : 'bg-gray-300'}`} />
      </div>
    </div>
  )
})

// Search Results Grid Skeleton
export const SearchGridSkeleton = memo(function SearchGridSkeleton({
  count = 12
}: {
  count?: number
}) {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-6">
      {[...Array(count)].map((_, i) => (
        <PoiCardSkeleton key={i} />
      ))}
    </div>
  )
})

// Loading Spinner Component
export const LoadingSpinner = memo(function LoadingSpinner({
  size = "default",
  className = ""
}: {
  size?: "small" | "default" | "large"
  className?: string
}) {
  const sizeClasses = {
    small: "w-4 h-4",
    default: "w-6 h-6", 
    large: "w-8 h-8"
  }

  return (
    <div className={`animate-spin rounded-full border-2 border-gray-300 border-t-gray-600 ${sizeClasses[size]} ${className}`} />
  )
})

// Full Page Loading Component
export const PageLoading = memo(function PageLoading({
  message = "Loading..."
}: {
  message?: string
}) {
  return (
    <div className="h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <LoadingSpinner size="large" className="mx-auto mb-4" />
        <p className="text-gray-600 text-sm">{message}</p>
      </div>
    </div>
  )
})

// Button Loading State
export const ButtonLoading = memo(function ButtonLoading({
  children,
  isLoading = false,
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  isLoading?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      className={`relative ${className} ${isLoading ? 'cursor-not-allowed opacity-70' : ''}`}
      disabled={isLoading}
      {...props}
    >
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <LoadingSpinner size="small" />
        </div>
      )}
      <span className={isLoading ? 'opacity-0' : 'opacity-100'}>
        {children}
      </span>
    </button>
  )
})