"use server"

import { createServerClient } from "@supabase/ssr"
import { cookies } from "next/headers"
import { redirect } from "next/navigation"

// Update the signIn function to handle redirects properly
export async function signIn(prevState: any, formData: FormData) {
  // Check if formData is valid
  if (!formData) {
    return { error: "Form data is missing" }
  }

  const email = formData.get("email")
  const password = formData.get("password")

  // Validate required fields
  if (!email || !password) {
    return { error: "Email and password are required" }
  }

  const cookieStore = cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options)
            })
          } catch (error) {
            // The `set` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  )

  try {
    const { error } = await supabase.auth.signInWithPassword({
      email: email.toString(),
      password: password.toString(),
    })

    if (error) {
      return { error: error.message }
    }

    // Return success instead of redirecting directly
    return { success: true }
  } catch (error) {
    console.error("Login error:", error)
    return { error: "An unexpected error occurred. Please try again." }
  }
}

// Update the signUp function to handle potential null formData and additional user metadata
export async function signUp(prevState: any, formData: FormData) {
  // Check if formData is valid
  if (!formData) {
    return { error: "Form data is missing" }
  }

  const email = formData.get("email")
  const password = formData.get("password")
  const confirmPassword = formData.get("confirmPassword")
  const firstName = formData.get("firstName")
  const lastName = formData.get("lastName")
  const username = formData.get("username")

  // Validate required fields
  if (!email || !password) {
    return { error: "Email and password are required" }
  }

  // Validate password confirmation
  if (password !== confirmPassword) {
    return { error: "Passwords do not match" }
  }

  const cookieStore = cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options)
            })
          } catch (error) {
            // The `set` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  )

  // Normalize Supabase sign-up errors into user-friendly messages
  const humanizeSignUpError = (error: any) => {
    const raw = (error?.message || "").toString()
    const msg = raw.toLowerCase()
    if (msg.includes("already registered") || error?.status === 422) {
      return "An account with this email already exists. Try logging in, or resend the verification email if you didn’t receive it."
    }
    return raw || "Could not sign up. Please try again."
  }

  try {
    const { data, error } = await supabase.auth.signUp({
      email: email.toString(),
      password: password.toString(),
      options: {
        emailRedirectTo:
          process.env.NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL ||
          `${process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"}/auth/callback`,
        data: {
          first_name: firstName?.toString(),
          last_name: lastName?.toString(),
          username: username?.toString(),
        },
      },
    })

    // Supabase sometimes returns no error but an empty identities array when the email already exists (confirmed)
    const identities = (data as any)?.user?.identities
    if (!error && Array.isArray(identities) && identities.length === 0) {
      return { error: "An account with this email already exists. Try logging in, or resend the verification email if you didn’t receive it." }
    }

    if (error) {
      return { error: humanizeSignUpError(error) }
    }

    return { success: "Check your email to confirm your account." }
  } catch (error) {
    console.error("Sign up error:", error)
    return { error: "An unexpected error occurred. Please try again." }
  }
}

export async function signInWithGoogle() {
  const cookieStore = cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options)
            })
          } catch (error) {
            // The `set` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  )

  try {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `https://roameo-app.vercel.app/auth/callback`,
      },
    })

    if (error) {
      return { error: error.message }
    }

    return { url: data.url }
  } catch (error) {
    console.error("Google sign-in error:", error)
    return { error: "An unexpected error occurred. Please try again." }
  }
}

export async function signOut() {
  const cookieStore = cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options)
            })
          } catch (error) {
            // The `set` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  )

  await supabase.auth.signOut()
  redirect("/auth/login")
}

// Single-argument versions for <form action={...}> in React 18/Next 14
export async function signInForm(formData: FormData) {
  const email = formData.get("email")?.toString()
  const password = formData.get("password")?.toString()
  if (!email || !password) {
    redirect(`/auth/login?error=${encodeURIComponent("Email and password are required")}`)
  }
  const cookieStore = cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options)
            })
          } catch (error) {
            // The `set` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  )
  const { error } = await supabase.auth.signInWithPassword({ email, password })
  if (error) {
    redirect(`/auth/login?error=${encodeURIComponent(error.message)}`)
  }
  redirect("/dashboard")
}

export async function signUpForm(formData: FormData) {
  const email = formData.get("email")?.toString()
  const password = formData.get("password")?.toString()
  const confirmPassword = formData.get("confirmPassword")?.toString()
  const firstName = formData.get("firstName")?.toString()
  const lastName = formData.get("lastName")?.toString()
  const username = formData.get("username")?.toString()

  if (!email || !password) {
    redirect(`/auth/login?error=${encodeURIComponent("Email and password are required")}`)
  }
  if (password !== confirmPassword) {
    redirect(`/auth/login?error=${encodeURIComponent("Passwords do not match")}`)
  }
  const cookieStore = cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options)
            })
          } catch (error) {
            // The `set` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  )
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo:
        process.env.NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL ||
        `${process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"}/auth/callback`,
      data: { first_name: firstName, last_name: lastName, username },
    },
  })
  // Detect existing user via identities array when no error is returned
  const identities = (data as any)?.user?.identities
  if (!error && Array.isArray(identities) && identities.length === 0) {
    const friendly = "An account with this email already exists. Try logging in, or resend the verification email if you didn’t receive it."
    redirect(`/auth/login?error=${encodeURIComponent(friendly)}`)
  }
  if (error) {
    const raw = (error.message || "").toString()
    const msg = raw.toLowerCase()
    const friendly = msg.includes("already registered") || (error as any)?.status === 422
      ? "An account with this email already exists. Try logging in, or resend the verification email if you didn’t receive it."
      : (raw || "Could not sign up. Please try again.")
    redirect(`/auth/login?error=${encodeURIComponent(friendly)}`)
  }
  // If confirmation is required, send to login with success message
  redirect(`/auth/login?success=${encodeURIComponent("Check your email to confirm your account.")}`)
}
