"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, CheckCircle2, KeyRound, Loader2, User } from "lucide-react";
import { listTrips, getSessionSettings, saveProviderCredential, updateSessionSettings } from "@/lib/api";
import { redirectToLogin } from "@/lib/auth-redirect";
import { supabase } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { EntranceMotion, SectionReveal } from "@/components/ui/site-motion";
import { toast } from "@/hooks/use-toast";
import type { SessionSettingsPayload } from "@/lib/types";

const styleOptions = [
  "relaxed",
  "balanced",
  "packed",
  "luxury",
  "budget",
  "family",
  "romantic",
  "adventure",
  "culture"
] as const;

const PROFILE_TOPOGRAPHY_DATA_URI =
  "data:image/svg+xml,%3Csvg width='1600' height='1200' viewBox='0 0 1600 1200' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cg stroke='%236b7280' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M154 165c32-14 77-19 125 10 35 21 58 52 67 79-60 4-120 1-179-7-6-28-11-55-13-82Z'/%3E%3Cpath d='M272 191c17-25 45-52 87-70 54 31 89 80 101 135-70 8-142 6-210-6 0-19 6-39 22-59Z'/%3E%3Cpath d='M1203 135c44 12 84 36 117 74-23 10-46 20-69 30-29-25-61-49-96-71 16-12 31-23 48-33Z'/%3E%3Cpath d='M1289 188c25 19 49 42 71 67-37 7-76 12-114 14-7-31-16-61-28-91 24-2 48 2 71 10Z'/%3E%3Cpath d='M1360 886c35-19 75-31 119-31-1 32-7 64-16 96-54-4-106-18-156-39 12-11 31-19 53-26Z'/%3E%3Cpath d='M1224 920c28-7 57-9 87-4-15 44-40 84-71 119-38-19-74-42-107-68 24-21 55-37 91-47Z'/%3E%3Cpath d='M245 931c43 0 86 9 126 27-18 28-39 54-61 79-50-14-97-35-141-61 20-17 47-31 76-40Z'/%3E%3Cpath d='M140 878c20 12 40 26 58 44-47 18-96 30-146 33-7-29-10-59-8-89 32-2 64 3 96 12Z'/%3E%3Cpath d='M528 1037c23-18 54-31 88-38 17 25 29 52 37 81-43 3-86 1-129-6-2-13-1-25 4-37Z'/%3E%3Cpath d='M863 979c25 12 47 32 65 60-36 11-73 19-110 24-8-21-20-41-35-59 26-16 54-25 80-25Z'/%3E%3Cpath d='M982 840c8-14 20-27 36-36 17 9 30 25 39 46-21 5-42 9-64 11-13-5-23-12-31-21 4 0 13 0 20 0Z'/%3E%3Cpath d='M328 510c18-10 40-11 61-3 7 17 7 35 0 53-20 4-41 5-61 3-11-13-13-34 0-53Z'/%3E%3Cpath d='M1170 479c22-8 47-6 69 8-6 18-18 35-34 48-20-6-39-15-57-26 4-12 11-22 22-30Z'/%3E%3Cpath d='M1467 214l34-42 25 34-34 41-25-33Z'/%3E%3Cpath d='M1479 205l10 15m-20 1 25-16'/%3E%3Cpath d='M230 800l18-28 22 25-17 28-23-25Z'/%3E%3Cpath d='M243 784l11 12m-18 5 23-15'/%3E%3Cpath d='M1038 170c16-30 36-56 64-75 22 18 40 41 54 67-40 5-79 8-118 8Z'/%3E%3Cpath d='M1062 129c10 3 18 8 24 16m18-15c-3 7-7 13-11 19'/%3E%3Cpath d='M690 103c23-33 54-61 90-81 22 29 37 63 43 100-51 5-103 3-153-4 2-5 9-10 20-15Z'/%3E%3Cpath d='M730 65c19 7 36 18 50 33m-3-47c-5 12-10 24-16 35'/%3E%3Cpath d='M1365 1035c18-44 48-82 86-108 26 28 46 61 57 97-47 20-96 32-146 36l3-25Z'/%3E%3Cpath d='M1432 959c9 8 17 18 22 29m24-38c-4 12-10 24-18 34'/%3E%3Cpath d='M51 1051c34-21 72-35 112-40 18 24 30 50 36 77-55 7-109 6-163-2 0-13 5-24 15-35Z'/%3E%3Cpath d='M106 1016c12 5 23 13 31 24m-4-35c-4 8-8 15-12 23'/%3E%3Cpath d='M502 242l7 9m12-2-12 7m-15 493 10 10m13-2-13 8m796-384 9 8m12-3-12 8m-1150-92 8 8m11-3-11 7m668 463 8 8m12-3-12 7'/%3E%3C/g%3E%3C/svg%3E";

const CARD_SHELL_CLASSNAME =
  "relative overflow-hidden rounded-[32px] border border-slate-100/60 bg-white/60 shadow-[0_20px_80px_rgba(15,23,42,0.08),_0_6px_20px_rgba(15,23,42,0.05)] backdrop-blur-2xl";

function CardTopography() {
  return (
    <>
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-[0.16]"
        style={{
          backgroundImage: `url("${PROFILE_TOPOGRAPHY_DATA_URI}")`,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
          backgroundSize: "1200px auto"
        }}
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.72),rgba(255,255,255,0.16)_42%,rgba(255,255,255,0.3)_100%)]"
      />
    </>
  );
}

export default function ProfilePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [tripCount, setTripCount] = useState(0);
  const [profileForm, setProfileForm] = useState({
    first_name: "",
    last_name: "",
    username: ""
  });
  const [settings, setSettings] = useState<SessionSettingsPayload>({
    providerSettings: {
      provider: "gemini",
      runMode: "balanced",
      keySource: "platform"
    },
    preferences: {
      currency: "INR",
      locale: "en-IN",
      styles: [],
      dietaryNotes: [],
      accessibilityNotes: []
    },
    credentials: [
      { provider: "gemini", keySource: "user", configured: false },
      { provider: "openai", keySource: "user", configured: false }
    ]
  });
  const [credentialDrafts, setCredentialDrafts] = useState<Record<string, string>>({
    gemini: "",
    openai: ""
  });
  const readOnlyEmail = user?.email || "";

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      const {
        data: { session }
      } = await supabase.auth.getSession();

      if (!session) {
        redirectToLogin();
        return;
      }

      const metadata = session.user.user_metadata || {};
      const name = (metadata.full_name || metadata.name || "").split(" ");
      const initialFirstName = metadata.first_name || name[0] || "";
      const initialLastName = metadata.last_name || name.slice(1).join(" ") || "";

      const [settingsResponse, tripsResponse] = await Promise.all([
        getSessionSettings(),
        listTrips()
      ]);

      if (!mounted) {
        return;
      }

      setUser(session.user);
      setProfileForm({
        first_name: initialFirstName,
        last_name: initialLastName,
        username: metadata.username || metadata.preferred_username || ""
      });
      setSettings(settingsResponse);
      setTripCount(tripsResponse.trips.length);
      setLoading(false);
    };

    void load();
    return () => {
      mounted = false;
    };
  }, [router]);

  const configuredProviders = useMemo(
    () => new Set(settings.credentials.filter((item) => item.configured).map((item) => item.provider)),
    [settings.credentials]
  );

  const toggleStyle = (style: (typeof styleOptions)[number]) => {
    setSettings((current) => {
      const styles = new Set(current.preferences.styles);
      if (styles.has(style)) {
        styles.delete(style);
      } else {
        styles.add(style);
      }

      return {
        ...current,
        preferences: {
          ...current.preferences,
          styles: Array.from(styles)
        }
      };
    });
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await supabase.auth.updateUser({ data: profileForm });
      await updateSessionSettings({
        providerSettings: settings.providerSettings,
        preferences: settings.preferences
      });

      const pendingCredentialSaves = Object.entries(credentialDrafts)
        .filter(([, value]) => value.trim())
        .map(([provider, apiKey]) =>
          saveProviderCredential(provider as "gemini" | "openai", apiKey.trim())
        );

      if (pendingCredentialSaves.length > 0) {
        await Promise.all(pendingCredentialSaves);
      }

      const refreshedSettings = await getSessionSettings();
      setSettings(refreshedSettings);
      setCredentialDrafts({ gemini: "", openai: "" });
      toast({
        title: "Settings saved",
        description: "Provider defaults, preferences, and profile details were updated."
      });
    } catch (error) {
      toast({
        title: "Could not save settings",
        description: error instanceof Error ? error.message : "Please try again.",
        variant: "destructive"
      });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f2f3f4]">
        <EntranceMotion className="rounded-[32px] border border-white/80 bg-white px-8 py-6 shadow-[0_24px_60px_rgba(34,74,187,0.12)]">
          <p className="font-roboto-mono text-sm uppercase tracking-[0.18em] text-[#5f74c8]">
            Loading profile
          </p>
          <h1 className="mt-2 text-xl font-semibold tracking-[-0.04em] text-gray-950">
            Restoring your travel preferences
          </h1>
        </EntranceMotion>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#f2f3f4] px-4 py-8 sm:px-6 lg:px-8">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-[0.18]"
        style={{
          backgroundImage: `url("${PROFILE_TOPOGRAPHY_DATA_URI}")`,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center top",
          backgroundSize: "cover"
        }}
      />
      <div className="mx-auto max-w-6xl">
        <EntranceMotion className="mb-8 flex items-center justify-between" delay={0.04}>
          <div className="flex items-center gap-3">
            <div className="relative flex h-8 w-8 items-center justify-center overflow-hidden rounded-full bg-black">
              <div className="h-2 w-2 rounded-full bg-white" />
            </div>
            <span className="text-[1.75rem] font-semibold tracking-[-0.05em] text-gray-950">roameo</span>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              className="rounded-full border-white/80 bg-white px-5 shadow-[0_14px_30px_rgba(15,23,42,0.06)] hover:bg-white"
              onClick={() => router.push("/dashboard")}
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to dashboard
            </Button>
            <Button
              variant="ghost"
              className="rounded-full bg-white px-5 text-gray-900 shadow-[0_14px_30px_rgba(15,23,42,0.06)] hover:bg-white"
              onClick={async () => {
                await supabase.auth.signOut();
                redirectToLogin();
              }}
            >
              Sign out
            </Button>
          </div>
        </EntranceMotion>

        <SectionReveal delay={0.08}>
          <div className="relative mb-8 overflow-hidden rounded-[40px] border border-white/70 bg-[linear-gradient(135deg,#eef4ff_0%,#d8e7ff_28%,#cddcff_54%,#e7f0ff_100%)] px-6 py-8 shadow-[0_24px_70px_rgba(76,107,184,0.12)] sm:px-8 lg:px-10 lg:py-10">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_24%_18%,rgba(255,255,255,0.92),rgba(255,255,255,0)_34%),radial-gradient(circle_at_82%_30%,rgba(122,145,255,0.16),rgba(122,145,255,0)_28%),radial-gradient(circle_at_54%_100%,rgba(255,255,255,0.8),rgba(255,255,255,0)_40%)]" />
            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.1)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.1)_1px,transparent_1px)] bg-[size:42px_42px] opacity-20" />
            <div className="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
              <div className="max-w-3xl">
                <p className="font-roboto-mono text-sm uppercase tracking-[0.24em] text-[#5f74c8]">
                  Profile and settings
                </p>
                <h1 className="mt-3 text-4xl font-semibold tracking-[-0.06em] text-[#101828] sm:text-5xl">
                  Keep Roameo aligned with how you actually travel
                </h1>
                <p className="mt-4 max-w-2xl text-base leading-7 text-[#475467]">
                  Tune your provider defaults, profile details, and travel preferences in the same dashboard visual language.
                </p>
              </div>
              <div className="w-full max-w-sm rounded-[28px] border border-white/80 bg-white/58 px-5 py-4 text-sm text-[#101828] shadow-[0_18px_50px_rgba(15,23,42,0.08)] backdrop-blur-md">
                <div className="font-roboto-mono text-[11px] uppercase tracking-[0.22em] text-[#6b7ed6]">
                  Workspace sync
                </div>
                <div className="mt-2 text-xl font-semibold tracking-[-0.04em] text-[#101828]">{tripCount} synced sessions</div>
                <div className="mt-2 break-all text-[#475467]">{user?.email}</div>
              </div>
            </div>
          </div>
        </SectionReveal>

        <div className="grid gap-6 lg:grid-cols-[1.05fr_1.35fr]">
          <SectionReveal delay={0.04}>
            <Card className={CARD_SHELL_CLASSNAME}>
            <CardTopography />
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2 text-gray-950">
                <User className="h-5 w-5 text-[#5f74c8]" />
                Account
              </CardTitle>
            </CardHeader>
            <CardContent className="relative space-y-5">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <Label htmlFor="first_name">First name</Label>
                  <Input
                    id="first_name"
                    value={profileForm.first_name}
                    onChange={(event) =>
                      setProfileForm((current) => ({
                        ...current,
                        first_name: event.target.value
                      }))
                    }
                    className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]"
                  />
                </div>
                <div>
                  <Label htmlFor="last_name">Last name</Label>
                  <Input
                    id="last_name"
                    value={profileForm.last_name}
                    onChange={(event) =>
                      setProfileForm((current) => ({
                        ...current,
                        last_name: event.target.value
                      }))
                    }
                    className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]"
                  />
                </div>
              </div>
              <div>
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  value={profileForm.username}
                  onChange={(event) =>
                    setProfileForm((current) => ({
                      ...current,
                      username: event.target.value
                    }))
                  }
                  className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]"
                />
              </div>
              <div>
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  value={readOnlyEmail}
                  readOnly
                  disabled
                  className="mt-2 rounded-2xl border-black/10 bg-[#f3f6fc] text-gray-500"
                />
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <Label>Currency</Label>
                  <Select
                    value={settings.preferences.currency}
                    onValueChange={(value) =>
                      setSettings((current) => ({
                        ...current,
                        preferences: {
                          ...current.preferences,
                          currency: value
                        }
                      }))
                    }
                  >
                    <SelectTrigger className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="INR">INR</SelectItem>
                      <SelectItem value="USD">USD</SelectItem>
                      <SelectItem value="EUR">EUR</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Locale</Label>
                  <Select
                    value={settings.preferences.locale}
                    onValueChange={(value) =>
                      setSettings((current) => ({
                        ...current,
                        preferences: {
                          ...current.preferences,
                          locale: value
                        }
                      }))
                    }
                  >
                    <SelectTrigger className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en-IN">English (India)</SelectItem>
                      <SelectItem value="en-US">English (US)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
            </Card>
          </SectionReveal>

          <SectionReveal delay={0.1}>
            <Card className={CARD_SHELL_CLASSNAME}>
            <CardTopography />
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2 text-gray-950">
                <KeyRound className="h-5 w-5 text-[#5f74c8]" />
                Agent runtime defaults
              </CardTitle>
            </CardHeader>
            <CardContent className="relative space-y-6">
              <div className="grid gap-4 sm:grid-cols-3">
                <div>
                  <Label>Provider</Label>
                  <Select
                    value={settings.providerSettings.provider}
                    onValueChange={(value) =>
                      setSettings((current) => ({
                        ...current,
                        providerSettings: {
                          ...current.providerSettings,
                          provider: value as "gemini" | "openai"
                        }
                      }))
                    }
                  >
                    <SelectTrigger className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gemini">Gemini</SelectItem>
                      <SelectItem value="openai">OpenAI</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Run mode</Label>
                  <Select
                    value={settings.providerSettings.runMode}
                    onValueChange={(value) =>
                      setSettings((current) => ({
                        ...current,
                        providerSettings: {
                          ...current.providerSettings,
                          runMode: value as "fast" | "balanced" | "deep"
                        }
                      }))
                    }
                  >
                    <SelectTrigger className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fast">Fast</SelectItem>
                      <SelectItem value="balanced">Balanced</SelectItem>
                      <SelectItem value="deep">Deep</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Key source</Label>
                  <Select
                    value={settings.providerSettings.keySource}
                    onValueChange={(value) =>
                      setSettings((current) => ({
                        ...current,
                        providerSettings: {
                          ...current.providerSettings,
                          keySource: value as "platform" | "user"
                        }
                      }))
                    }
                  >
                    <SelectTrigger className="mt-2 rounded-2xl border-black/10 bg-[#fbfcff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="platform">Platform managed</SelectItem>
                      <SelectItem value="user">Bring your own key</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label>Travel styles</Label>
                <div className="mt-3 flex flex-wrap gap-2">
                  {styleOptions.map((style) => {
                    const active = settings.preferences.styles.includes(style);
                    return (
                      <Button
                        key={style}
                        type="button"
                        variant={active ? "default" : "outline"}
                        className={`rounded-full ${
                          active
                            ? "bg-black text-white hover:bg-gray-800"
                            : "border-black/10 bg-[#fbfcff] text-gray-700 hover:bg-white"
                        }`}
                        onClick={() => toggleStyle(style)}
                      >
                        {style}
                      </Button>
                    );
                  })}
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                {(["gemini", "openai"] as const).map((provider) => (
                  <div
                    key={provider}
                    className="rounded-[24px] border border-black/5 bg-white/72 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.04)] backdrop-blur-[2px]"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div>
                          <div className="font-medium text-gray-950">
                            {provider === "gemini" ? "Gemini" : "ChatGPT"}
                          </div>
                          <div className="text-xs text-gray-500">
                            {provider === "gemini" ? "Google Gemini API key" : "OpenAI API key"}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-500">
                        <CheckCircle2
                          className={`h-4 w-4 ${
                            configuredProviders.has(provider)
                              ? "text-emerald-600"
                              : "text-black/20"
                          }`}
                        />
                        {configuredProviders.has(provider)
                          ? "User key saved"
                          : "Using platform key"}
                      </div>
                    </div>
                    <Input
                      type="password"
                      placeholder={`Paste ${provider} API key`}
                      value={credentialDrafts[provider]}
                      onChange={(event) =>
                        setCredentialDrafts((current) => ({
                          ...current,
                          [provider]: event.target.value
                        }))
                      }
                      className="mt-3 rounded-2xl border-black/10 bg-white"
                    />
                  </div>
                ))}
              </div>

              <div className="flex justify-end">
                <Button
                  onClick={() => {
                    void handleSave();
                  }}
                  disabled={saving}
                  className="rounded-full bg-[#1f1b16] px-6 text-white hover:bg-[#352c24]"
                >
                  {saving ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving
                    </>
                  ) : (
                    "Save settings"
                  )}
                </Button>
              </div>
            </CardContent>
            </Card>
          </SectionReveal>
        </div>
      </div>
    </div>
  );
}
