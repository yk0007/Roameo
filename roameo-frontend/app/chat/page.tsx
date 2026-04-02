import { redirect } from "next/navigation";
import ChatPageClient from "./chat-page-client";
import { createClient } from "@/lib/supabase/server";

export default async function ChatPage() {
  const supabase = await createClient();
  const {
    data: { session }
  } = await supabase.auth.getSession();

  if (!session) {
    redirect("/auth/login");
  }

  return <ChatPageClient />;
}
