import { useEffect, useRef, useState } from "react";
import type { HealthResponse } from "../types";

export function useHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchHealth = async () => {
    try {
      const res = await fetch("/health");
      const data: HealthResponse = await res.json();
      setHealth(data);
      setError(null);
    } catch {
      setError("Cannot reach service");
    }
  };

  useEffect(() => {
    fetchHealth();
    timerRef.current = setInterval(fetchHealth, intervalMs);
    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, [intervalMs]);

  return { health, error, refresh: fetchHealth };
}
