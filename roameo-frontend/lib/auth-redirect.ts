export function redirectToLogin() {
  if (typeof window !== "undefined") {
    window.location.replace("/auth/login");
    return;
  }
}
