import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Toaster } from "@/components/ui/toaster"

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
})

export const metadata: Metadata = {
  title: "Roameo - AI Travel Planner",
  description: "Multi-agent AI travel planner that creates personalized, budget-aware itineraries",
  generator: "Roameo",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${inter.variable}`}>
      <body
        className="antialiased font-sans"
        style={{ fontFamily: "var(--font-inter), ui-sans-serif, system-ui, sans-serif" }}
      >
        {children}
        <Toaster />
      </body>
    </html>
  )
}
