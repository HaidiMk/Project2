import { useState, useEffect, useCallback } from "react";
import { fetchDashboardData } from "../services/api.js";

export function useDashboardData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setUnavailable(false);
    try {
      const result = await fetchDashboardData();
      setData(result);
    } catch {
      setData(null);
      setUnavailable(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, unavailable, reload: load };
}
