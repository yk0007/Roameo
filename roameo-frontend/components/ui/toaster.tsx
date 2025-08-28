"use client"

import { useToast } from "@/hooks/use-toast"
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast"
import { CheckCircle, AlertCircle, XCircle, Info } from "lucide-react"

function getToastIcon(variant?: string) {
  switch (variant) {
    case "success":
      return <CheckCircle className="w-5 h-5 flex-shrink-0" />
    case "destructive":
      return <XCircle className="w-5 h-5 flex-shrink-0" />
    case "warning":
      return <AlertCircle className="w-5 h-5 flex-shrink-0" />
    case "info":
      return <Info className="w-5 h-5 flex-shrink-0" />
    default:
      return <Info className="w-5 h-5 flex-shrink-0" />
  }
}

export function Toaster() {
  const { toasts } = useToast()

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, variant, ...props }) {
        const icon = getToastIcon(variant)
        
        return (
          <Toast key={id} variant={variant} {...props}>
            <div className="flex items-start gap-3 w-full">
              {/* Icon */}
              <div className="mt-0.5">
                {icon}
              </div>
              
              {/* Content */}
              <div className="flex-1 grid gap-1">
                {title && <ToastTitle>{title}</ToastTitle>}
                {description && (
                  <ToastDescription>{description}</ToastDescription>
                )}
              </div>
              
              {/* Action */}
              {action && (
                <div className="flex-shrink-0">
                  {action}
                </div>
              )}
            </div>
            <ToastClose />
          </Toast>
        )
      })}
      <ToastViewport />
    </ToastProvider>
  )
}
