import { redirect } from "next/navigation";
import ProfilePageClient from "./profile-page-client";
import { createClient } from "@/lib/supabase/server";

export default async function ProfilePage() {
  const supabase = await createClient();
  const {
    data: { session }
  } = await supabase.auth.getSession();

  if (!session) {
    redirect("/auth/login");
  }

  return <ProfilePageClient />;
}
