// Single source of truth for the Django API base URL
// and the future Admin/Viewer Dashboard endpoint.

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// STATUS: Awaiting Backend-team implementation.
// The Django Backend team will confirm the final endpoint later.
const DASHBOARD_STATS_PATH = "/api/dashboard/stats/";

export const ENDPOINTS = {
  dashboardStats: `${API_BASE_URL}${DASHBOARD_STATS_PATH}`,
};
