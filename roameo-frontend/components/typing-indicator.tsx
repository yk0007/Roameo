"use client"

import { AgenticStatus } from "./agentic-status"

interface TypingIndicatorProps {
  isVisible: boolean
}

export function TypingIndicator({ isVisible }: TypingIndicatorProps) {
  if (!isVisible) return null

  return <AgenticStatus mode="general" />
}
