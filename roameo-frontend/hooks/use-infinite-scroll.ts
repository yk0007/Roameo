"use client"

import { useEffect, useRef, useCallback } from "react"

interface UseInfiniteScrollOptions {
  hasMore: boolean
  isLoading: boolean
  onLoadMore: () => void
  threshold?: number
}

export function useInfiniteScroll({
  hasMore,
  isLoading,
  onLoadMore,
  threshold = 200
}: UseInfiniteScrollOptions) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef(false)

  const handleScroll = useCallback(() => {
    const element = scrollRef.current
    if (!element || loadingRef.current || !hasMore || isLoading) return

    const { scrollTop, scrollHeight, clientHeight } = element
    const isNearBottom = scrollHeight - scrollTop - clientHeight < threshold

    if (isNearBottom) {
      loadingRef.current = true
      onLoadMore()
      // Reset loading flag after a delay to prevent rapid calls
      setTimeout(() => {
        loadingRef.current = false
      }, 500)
    }
  }, [hasMore, isLoading, onLoadMore, threshold])

  useEffect(() => {
    const element = scrollRef.current
    if (!element) return

    element.addEventListener('scroll', handleScroll, { passive: true })
    return () => element.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  return scrollRef
}
