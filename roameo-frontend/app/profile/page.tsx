import type { Metadata } from "next";
import { redirect } from "next/navigation";
import ProfilePageClient from "./profile-page-client";
import { createClient } from "@/lib/supabase/server";

export const metadata: Metadata = {
  title: "Profile Settings",
};

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
