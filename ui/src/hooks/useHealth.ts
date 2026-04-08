import { useCallback, useEffect, useRef, useState } from "react";
import type { HealthResponse } from "../types";

export function useHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/health");
      const data: HealthResponse = await res.json();
      setHealth(data);
      setError(null);
    } catch {
      setError("Cannot reach service");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    timerRef.current = setInterval(fetchHealth, intervalMs);
    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, [fetchHealth, intervalMs]);

  return { health, loading, error, refresh: fetchHealth };
}
