import { useState, useEffect, useCallback } from "react";
import { fetchDashboardData } from "../services/api.js";

/**
 * Loading/data state around the future Admin/Viewer Dashboard API.
 * Any failure — including the endpoint not existing yet, which is the
 * expected current state — is treated as "no data available" rather than
 * a hard error, so the page degrades to neutral empty states instead of
 * looking broken.
 */
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
