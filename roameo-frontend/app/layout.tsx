import type React from "react"
import type { Metadata } from "next"
import { Inter, Roboto_Mono, Pacifico, Prompt } from "next/font/google"
import "./globals.css"
import { Toaster } from "@/components/ui/toaster"
import { FloatingNavbarProvider } from "@/components/floating-navbar-provider"

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
})

const robotoMono = Roboto_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-roboto-mono",
})

const pacifico = Pacifico({
  weight: "400",
  subsets: ["latin"],
  display: "swap",
  variable: "--font-pacifico",
})

const prompt = Prompt({
  weight: ["400", "500"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-prompt",
})

export const metadata: Metadata = {
  title: "Roameo - AI Travel Planner",
  description: "Multi-agent AI travel planner that creates personalized, budget-aware itineraries",
  generator: "Roameo",
  icons: {
    icon: [
      {
        url: '/favicon.svg',
        type: 'image/svg+xml',
      }
    ],
    shortcut: '/favicon.svg',
    apple: '/apple-touch-icon.svg'
  }
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${robotoMono.variable} ${pacifico.variable} ${prompt.variable}`}>
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="shortcut icon" href="/favicon.svg" />
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#0f172a" />
      </head>
      <body
        className="antialiased font-sans"
        style={{ fontFamily: "var(--font-inter), ui-sans-serif, system-ui, sans-serif" }}
      >
        <FloatingNavbarProvider>
          {children}
        </FloatingNavbarProvider>
        <Toaster />
      </body>
    </html>
  )
}
