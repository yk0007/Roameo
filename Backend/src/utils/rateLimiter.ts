type Key = string;

export class SimpleRateLimiter {
  private hits = new Map<Key, { count: number; resetAt: number }>();
  constructor(private limit: number, private intervalMs: number) {}

  allow(key: Key): boolean {
    const now = Date.now();
    const rec = this.hits.get(key);
    if (!rec || now >= rec.resetAt) {
      this.hits.set(key, { count: 1, resetAt: now + this.intervalMs });
      return true;
    }
    if (rec.count < this.limit) {
      rec.count++;
      return true;
    }
    return false;
  }
}
