"use client"

import { useState } from "react"
import { useFormStatus } from "react-dom"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { Eye, EyeOff, Loader2 } from "lucide-react"
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
  const [isSignUp, setIsSignUp] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [googlePending, setGooglePending] = useState(false)

  const currentAction = isSignUp ? signUpForm : signInForm

  const handleGoogleSignIn = async () => {
    setGooglePending(true)
    const result = await signInWithGoogle()
    if (result.url) {
      window.location.href = result.url
      return
    }
    setGooglePending(false)
  }

  return (
    <div className="w-full space-y-6">
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-black shadow-[0_10px_24px_rgba(0,0,0,0.16)]">
            <div className="h-2 w-2 rounded-full bg-white"></div>
          </div>
          <span className="text-[1.7rem] font-semibold tracking-[-0.05em] text-slate-950">roameo</span>
        </div>

        <div className="inline-flex rounded-full border border-slate-200 bg-slate-50 p-1">
          <button
            type="button"
            onClick={() => setIsSignUp(false)}
            className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
              !isSignUp ? "bg-black text-white shadow-sm" : "text-slate-500 hover:text-slate-950"
            }`}
          >
            Log in
          </button>
          <button
            type="button"
            onClick={() => setIsSignUp(true)}
            className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
              isSignUp ? "bg-black text-white shadow-sm" : "text-slate-500 hover:text-slate-950"
            }`}
          >
            Create account
          </button>
        </div>
      </div>

      <form action={currentAction} className="space-y-5">
        <AnimatePresence mode="wait">
          <motion.div
            key={isSignUp ? "signup-fields" : "signin-fields"}
            initial={{ opacity: 0, x: isSignUp ? 50 : -50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: isSignUp ? -50 : 50 }}
            transition={{ duration: 0.3 }}
            className="space-y-3.5"
          >
            {isSignUp && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label htmlFor="firstName" className="block text-sm font-medium text-slate-700">
                      First Name
                    </label>
                    <Input
                      id="firstName"
                      name="firstName"
                      type="text"
                      placeholder="John"
                      required
                      className="h-11 rounded-2xl border-slate-200 bg-slate-50 px-4 text-slate-950 placeholder:text-slate-400 focus:border-slate-300 focus:ring-slate-200"
                    />
                  </div>
                  <div className="space-y-1">
                    <label htmlFor="lastName" className="block text-sm font-medium text-slate-700">
                      Last Name
                    </label>
                    <Input
                      id="lastName"
                      name="lastName"
                      type="text"
                      placeholder="Doe"
                      required
                      className="h-11 rounded-2xl border-slate-200 bg-slate-50 px-4 text-slate-950 placeholder:text-slate-400 focus:border-slate-300 focus:ring-slate-200"
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <label htmlFor="username" className="block text-sm font-medium text-slate-700">
                    Username
                  </label>
                  <Input
                    id="username"
                    name="username"
                    type="text"
                    placeholder="Choose a username"
                    required
                    className="h-11 rounded-2xl border-slate-200 bg-slate-50 px-4 text-slate-950 placeholder:text-slate-400 focus:border-slate-300 focus:ring-slate-200"
                  />
                </div>
              </>
            )}

            {/* Email Field */}
            <div className="space-y-1">
              <label htmlFor="email" className="block text-sm font-medium text-slate-700">
                Email
              </label>
              <Input
                id="email"
                name="email"
                type="email"
                placeholder="name@example.com"
                required
                className="h-11 rounded-2xl border-slate-200 bg-slate-50 px-4 text-slate-950 placeholder:text-slate-400 focus:border-slate-300 focus:ring-slate-200"
              />
            </div>

            {/* Password Field */}
            <div className="space-y-1">
              <label htmlFor="password" className="block text-sm font-medium text-slate-700">
                Password
              </label>
              <div className="relative">
                <Input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  placeholder={isSignUp ? "Create a password" : "Enter your password"}
                  required
                  minLength={8}
                  className="h-11 rounded-2xl border-slate-200 bg-slate-50 px-4 pr-10 text-slate-950 placeholder:text-slate-400 focus:border-slate-300 focus:ring-slate-200"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((current) => !current)}
                  className="absolute inset-y-0 right-0 flex items-center pr-4 text-slate-400 transition-colors hover:text-slate-600"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {isSignUp && (
              <div className="space-y-1">
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-slate-700">
                  Confirm Password
                </label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="Repeat your password"
                    required
                    minLength={8}
                    className="h-11 rounded-2xl border-slate-200 bg-slate-50 px-4 pr-10 text-slate-950 placeholder:text-slate-400 focus:border-slate-300 focus:ring-slate-200"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword((current) => !current)}
                    className="absolute inset-y-0 right-0 flex items-center pr-4 text-slate-400 transition-colors hover:text-slate-600"
                    aria-label={showConfirmPassword ? "Hide confirm password" : "Show confirm password"}
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
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
                className="border-slate-300"
              />
              <label htmlFor="remember-me" className="text-sm text-slate-700">
                Remember me
              </label>
            </div>
            <button
              type="button"
              className="text-sm text-slate-500 transition-colors hover:text-slate-950"
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
          disabled={googlePending}
          className="h-11 w-full rounded-2xl border-slate-200 bg-white text-slate-700 transition-colors hover:bg-slate-50 disabled:opacity-70"
        >
          {googlePending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Redirecting to Google...
            </>
          ) : (
            <>
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
            </>
          )}
        </Button>

        {/* Switch between Sign In and Sign Up */}
        <div className="text-center">
          <span className="text-sm text-slate-600">
            {isSignUp ? "Already have an account?" : "New to roameo?"}
          </span>{" "}
          <button
            type="button"
            onClick={() => setIsSignUp(!isSignUp)}
            className="text-sm font-medium text-slate-950 transition-colors hover:text-slate-700"
          >
            {isSignUp ? "Sign in" : "Register now"}
          </button>
        </div>
      </form>
    </div>
  )
}
