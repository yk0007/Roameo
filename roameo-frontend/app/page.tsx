import type { Metadata } from "next"
import { LandingPage } from "@/components/landing-page"

export const metadata: Metadata = {
  title: "AI Travel Planner",
}

export default function HomePage() {
  return <LandingPage />
}
