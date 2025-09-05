"use client"

import { useEffect } from "react"
import { usePathname, useRouter, useSearchParams } from "next/navigation"

/**
 * Clears specified query params from the URL on mount using router.replace
 * without causing a full navigation. Useful for one-time banners (success/error).
 */
export default function ClearQuery({ keys = ["success"] }: { keys?: string[] }) {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  useEffect(() => {
    if (!searchParams) return
    const params = new URLSearchParams(searchParams.toString())
    let changed = false
    for (const k of keys) {
      if (params.has(k)) {
        params.delete(k)
        changed = true
      }
    }
    if (changed) {
      const search = params.toString()
      const url = search ? `${pathname}?${search}` : pathname
      router.replace(url)
    }
  }, [router, pathname, searchParams, keys])

  return null
}
