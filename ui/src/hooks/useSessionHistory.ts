import { useCallback, useEffect, useState } from "react";
import type { ChatSession } from "../types";

const STORAGE_KEY = "llm-playground-sessions";
const MAX_SESSIONS = 50;

function load(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as ChatSession[];
  } catch {
    return [];
  }
}

function save(sessions: ChatSession[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    /* storage full — ignore */
  }
}

export function useSessionHistory() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => load());

  useEffect(() => {
    save(sessions);
  }, [sessions]);

  const upsertSession = useCallback((session: ChatSession) => {
    setSessions((prev) => {
      const idx = prev.findIndex((s) => s.id === session.id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = session;
        return next;
      }
      // Prepend new session, cap list length
      return [session, ...prev].slice(0, MAX_SESSIONS);
    });
  }, []);

  const deleteSession = useCallback((id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
  }, []);

  const clearAll = useCallback(() => setSessions([]), []);

  return { sessions, upsertSession, deleteSession, clearAll };
}
