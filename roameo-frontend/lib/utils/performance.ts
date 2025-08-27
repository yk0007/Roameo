import { useCallback, useRef, useEffect } from 'react'

// Debounce hook for delaying function execution
export function useDebounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): T {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const debouncedFunc = useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      timeoutRef.current = setTimeout(() => {
        func(...args)
      }, delay)
    },
    [func, delay]
  ) as T

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  return debouncedFunc
}

// Throttle hook for limiting function execution frequency
export function useThrottle<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): T {
  const lastCallRef = useRef<number>(0)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const throttledFunc = useCallback(
    (...args: Parameters<T>) => {
      const now = Date.now()
      const timeSinceLastCall = now - lastCallRef.current

      if (timeSinceLastCall >= delay) {
        lastCallRef.current = now
        func(...args)
      } else {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
        }
        timeoutRef.current = setTimeout(() => {
          lastCallRef.current = Date.now()
          func(...args)
        }, delay - timeSinceLastCall)
      }
    },
    [func, delay]
  ) as T

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  return throttledFunc
}

// Scroll throttling for smooth scrolling experiences
export function useScrollThrottle<T extends (...args: any[]) => any>(
  func: T,
  delay: number = 16 // ~60fps
): T {
  return useThrottle(func, delay)
}

// Search input debouncing
export function useSearchDebounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number = 300
): T {
  return useDebounce(func, delay)
}

// Window resize throttling
export function useResizeThrottle<T extends (...args: any[]) => any>(
  func: T,
  delay: number = 100
): T {
  return useThrottle(func, delay)
}