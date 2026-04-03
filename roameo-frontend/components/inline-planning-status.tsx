"use client"

import { AgenticStatus } from "./agentic-status"

interface InlinePlanningStatusProps {
  isVisible: boolean
  onComplete?: () => void
}

export function InlinePlanningStatus({ isVisible }: InlinePlanningStatusProps) {
  if (!isVisible) return null

  return <AgenticStatus mode="planning" />
}
