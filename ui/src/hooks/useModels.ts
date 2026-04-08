import { useCallback, useEffect, useRef, useState } from "react";
import type { ModelsResponse } from "../types";

export function useModels(intervalMs = 10000) {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/models");
      const data: ModelsResponse = await res.json();
      setModels(data);
      setError(null);
    } catch {
      setError("Failed to fetch models");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    timerRef.current = setInterval(fetchModels, intervalMs);
    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, [fetchModels, intervalMs]);

  return { models, loading, error, refresh: fetchModels };
}
