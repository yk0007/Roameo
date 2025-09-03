"use client"

import { useState } from "react"
import { useFormStatus } from "react-dom"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { Loader2 } from "lucide-react"
import { useRouter } from "next/navigation"
import { signInForm, signUpForm, signInWithGoogle } from "@/lib/actions"
import { motion, AnimatePresence } from "framer-motion"

function SubmitButton({ isSignUp }: { isSignUp: boolean }) {
  const { pending } = useFormStatus()

  return (
    <Button
      type="submit"
      disabled={pending}
      className="w-full bg-gray-800 hover:bg-gray-900 text-white py-3 text-sm font-medium rounded-lg transition-colors"
    >
      {pending ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          {isSignUp ? "Creating account..." : "Signing in..."}
        </>
      ) : isSignUp ? (
        "Create Account"
      ) : (
        "Log in"
      )}
    </Button>
  )
}

export default function AuthForm() {
  const router = useRouter()
  const [isSignUp, setIsSignUp] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)

  const currentAction = isSignUp ? signUpForm : signInForm

  const handleGoogleSignIn = async () => {
    const result = await signInWithGoogle()
    if (result.url) {
      window.location.href = result.url
    }
  }

  return (
    <div className="w-full space-y-6">
      {/* Logo and Brand */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-6">
          <div className="w-6 h-6 bg-black rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
          </div>
          <span className="text-lg font-semibold text-gray-900">roameo</span>
        </div>
      </div>

      {/* Title and Description */}
      <div className="space-y-2 mb-8">
        <AnimatePresence mode="wait">
          <motion.h1
            key={isSignUp ? "signup" : "signin"}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="text-2xl font-semibold text-gray-900"
          >
            {isSignUp ? "Create Account" : "Welcome back,"}
          </motion.h1>
        </AnimatePresence>
        <p className="text-sm text-gray-600">
          {isSignUp ? "Join Roameo to start planning your adventures" : "Please enter your details"}
        </p>
      </div>

      {/* Form */}
      <form action={currentAction} className="space-y-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={isSignUp ? "signup-fields" : "signin-fields"}
            initial={{ opacity: 0, x: isSignUp ? 50 : -50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: isSignUp ? -50 : 50 }}
            transition={{ duration: 0.3 }}
            className="space-y-4"
          >
            {isSignUp && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label htmlFor="firstName" className="block text-sm font-medium text-gray-700">
                      First Name
                    </label>
                    <Input
                      id="firstName"
                      name="firstName"
                      type="text"
                      placeholder="John"
                      required
                      className="border-gray-300 focus:border-gray-500 focus:ring-gray-500 rounded-lg placeholder:text-gray-400"
                    />
                  </div>
                  <div className="space-y-1">
                    <label htmlFor="lastName" className="block text-sm font-medium text-gray-700">
                      Last Name
                    </label>
                    <Input
                      id="lastName"
                      name="lastName"
                      type="text"
                      placeholder="Doe"
                      required
                      className="border-gray-300 focus:border-gray-500 focus:ring-gray-500 rounded-lg placeholder:text-gray-400"
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                    Username
                  </label>
                  <Input
                    id="username"
                    name="username"
                    type="text"
                    placeholder="johndoe"
                    required
                    className="border-gray-300 focus:border-gray-500 focus:ring-gray-500 rounded-lg"
                  />
                </div>
              </>
            )}

            {/* Email Field */}
            <div className="space-y-1">
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email
              </label>
              <Input
                id="email"
                name="email"
                type="email"
                placeholder={""}
                required
                className="border-gray-300 focus:border-gray-500 focus:ring-gray-500 rounded-lg"
              />
            </div>

            {/* Password Field */}
            <div className="space-y-1">
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <Input
                id="password"
                name="password"
                type="password"
                placeholder={""}
                required
                className="border-gray-300 focus:border-gray-500 focus:ring-gray-500 rounded-lg"
              />
            </div>

            {isSignUp && (
              <div className="space-y-1">
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                  Confirm Password
                </label>
                <Input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  placeholder={""}
                  required
                  className="border-gray-300 focus:border-gray-500 focus:ring-gray-500 rounded-lg"
                />
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Remember me and Forgot password (only for sign in) */}
        {!isSignUp && (
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Checkbox 
                id="remember-me" 
                checked={rememberMe}
                onCheckedChange={(checked) => setRememberMe(checked as boolean)}
                className="border-gray-300"
              />
              <label htmlFor="remember-me" className="text-sm text-gray-700">
                Remember me
              </label>
            </div>
            <button
              type="button"
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Forgot password?
            </button>
          </div>
        )}

        {/* Submit Button */}
        <SubmitButton isSignUp={isSignUp} />

        {/* Google Sign In Button */}
        <Button
          type="button"
          onClick={handleGoogleSignIn}
          variant="outline"
          className="w-full border-gray-300 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
        >
          <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
            <path
              fill="#4285F4"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="#34A853"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="#FBBC05"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
            />
            <path
              fill="#EA4335"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
            />
          </svg>
          Continue with Google
        </Button>

        {/* Switch between Sign In and Sign Up */}
        <div className="text-center">
          <span className="text-sm text-gray-600">
            {isSignUp ? "Already have an account?" : "New to roameo?"}
          </span>{" "}
          <button
            type="button"
            onClick={() => setIsSignUp(!isSignUp)}
            className="text-sm text-gray-900 hover:text-gray-700 font-medium transition-colors"
          >
            {isSignUp ? "Sign in" : "Register now"}
          </button>
        </div>
      </form>
    </div>
  )
}
