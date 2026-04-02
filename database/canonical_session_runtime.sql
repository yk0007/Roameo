-- Canonical Roameo runtime schema
-- Apply after backing up any legacy chat_sessions/messages tables.

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.travel_sessions (
  id text primary key,
  user_id uuid references auth.users(id) on delete cascade,
  title text not null default 'Untitled trip',
  provider_settings jsonb not null default '{"provider":"gemini","runMode":"balanced","keySource":"platform"}'::jsonb,
  memory jsonb not null default '{}'::jsonb,
  destination_summary text,
  total_days integer,
  current_plan_version integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.session_messages (
  id text primary key,
  session_id text not null references public.travel_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system', 'tool')),
  phase text check (phase in ('thinking', 'tooling', 'draft', 'final')),
  content text not null,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.session_plan_snapshots (
  id text primary key,
  session_id text not null references public.travel_sessions(id) on delete cascade,
  version integer not null check (version >= 0),
  snapshot jsonb not null,
  created_at timestamptz not null default now(),
  unique (session_id, version)
);

create table if not exists public.session_poi_catalogs (
  session_id text primary key references public.travel_sessions(id) on delete cascade,
  catalog jsonb not null default '{"version":1,"items":{}}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists public.session_saved_pois (
  session_id text not null references public.travel_sessions(id) on delete cascade,
  poi_id text not null,
  created_at timestamptz not null default now(),
  primary key (session_id, poi_id)
);

create table if not exists public.session_agent_traces (
  id text primary key,
  session_id text not null references public.travel_sessions(id) on delete cascade,
  turn_id text not null,
  agent text not null,
  status text not null check (status in ('queued', 'running', 'completed', 'failed')),
  label text not null,
  detail text,
  created_at timestamptz not null default now()
);

create table if not exists public.user_provider_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  provider_settings jsonb not null default '{"provider":"gemini","runMode":"balanced","keySource":"platform"}'::jsonb,
  preferences jsonb not null default '{"currency":"INR","locale":"en-IN","styles":[],"dietaryNotes":[],"accessibilityNotes":[]}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists public.user_provider_credentials (
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null check (provider in ('gemini', 'openai')),
  key_source text not null check (key_source in ('platform', 'user')),
  encrypted_key text not null,
  updated_at timestamptz not null default now(),
  primary key (user_id, provider, key_source)
);

create index if not exists travel_sessions_user_updated_idx
  on public.travel_sessions (user_id, updated_at desc);

create index if not exists session_messages_session_created_idx
  on public.session_messages (session_id, created_at asc);

create index if not exists session_plan_snapshots_session_version_idx
  on public.session_plan_snapshots (session_id, version desc);

create index if not exists session_saved_pois_session_idx
  on public.session_saved_pois (session_id);

create index if not exists session_agent_traces_session_created_idx
  on public.session_agent_traces (session_id, created_at asc);

create trigger travel_sessions_set_updated_at
before update on public.travel_sessions
for each row execute function public.set_updated_at();

create trigger session_poi_catalogs_set_updated_at
before update on public.session_poi_catalogs
for each row execute function public.set_updated_at();

create trigger user_provider_settings_set_updated_at
before update on public.user_provider_settings
for each row execute function public.set_updated_at();

alter table public.travel_sessions enable row level security;
alter table public.session_messages enable row level security;
alter table public.session_plan_snapshots enable row level security;
alter table public.session_poi_catalogs enable row level security;
alter table public.session_saved_pois enable row level security;
alter table public.session_agent_traces enable row level security;
alter table public.user_provider_settings enable row level security;
alter table public.user_provider_credentials enable row level security;

create policy "travel_sessions_owner_all"
on public.travel_sessions
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "session_messages_owner_all"
on public.session_messages
for all
using (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_messages.session_id
      and sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_messages.session_id
      and sessions.user_id = auth.uid()
  )
);

create policy "session_plan_snapshots_owner_all"
on public.session_plan_snapshots
for all
using (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_plan_snapshots.session_id
      and sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_plan_snapshots.session_id
      and sessions.user_id = auth.uid()
  )
);

create policy "session_poi_catalogs_owner_all"
on public.session_poi_catalogs
for all
using (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_poi_catalogs.session_id
      and sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_poi_catalogs.session_id
      and sessions.user_id = auth.uid()
  )
);

create policy "session_saved_pois_owner_all"
on public.session_saved_pois
for all
using (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_saved_pois.session_id
      and sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_saved_pois.session_id
      and sessions.user_id = auth.uid()
  )
);

create policy "session_agent_traces_owner_all"
on public.session_agent_traces
for all
using (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_agent_traces.session_id
      and sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.travel_sessions sessions
    where sessions.id = session_agent_traces.session_id
      and sessions.user_id = auth.uid()
  )
);

create policy "user_provider_settings_owner_all"
on public.user_provider_settings
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "user_provider_credentials_owner_all"
on public.user_provider_credentials
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
