export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const DASHBOARD_STATS_PATH = "/api/dashboard/stats/";

export const ENDPOINTS = {
  dashboardStats: `${API_BASE_URL}${DASHBOARD_STATS_PATH}`,
};
