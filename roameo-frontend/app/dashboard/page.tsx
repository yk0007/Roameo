import { redirect } from "next/navigation";
import DashboardPageClient from "./dashboard-page-client";
import { createClient } from "@/lib/supabase/server";

export default async function DashboardPage() {
  const supabase = await createClient();
  const {
    data: { session }
  } = await supabase.auth.getSession();

  if (!session) {
    redirect("/auth/login");
  }

  return <DashboardPageClient />;
}
