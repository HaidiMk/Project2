import { ENDPOINTS } from "../config/api.js";

/**
 * fetchDashboardData()
 * ====================
 * Calls the future Admin/Viewer Dashboard API.
 *
 * STATUS: Awaiting Backend-team implementation. No endpoint exists yet at
 * ENDPOINTS.dashboardStats — calling this today is expected to fail (404 or
 * network error). The caller (useDashboardData) treats any failure as
 * "no data yet" and falls back to neutral empty states.
 *
 * Provisional response contract (backend team may adjust):
 * {
 *   "stats": {
 *     "total_recipes": number,
 *     "supported_conditions": number,
 *     "supported_allergies": number,
 *     "recommendations_count": number
 *   },
 *   "results": [
 *     {
 *       "name": string,
 *       "final_score": number,
 *       "calories": number|null,
 *       "protein": number|null,
 *       "ai_health_score": number|null,
 *       "expert_score": number|null
 *     }
 *   ],
 *   "chart_data": [
 *     { "label": string, "value": number }
 *   ]
 * }
 */
class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function parseJsonSafely(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function fetchDashboardData() {
  let response;
  try {
    response = await fetch(ENDPOINTS.dashboardStats);
  } catch {
    throw new ApiError("Cannot reach the Django API.", 0);
  }

  const data = await parseJsonSafely(response);

  if (!response.ok) {
    const detail =
      (data && data.detail) ||
      `Dashboard API request failed with status ${response.status}.`;
    throw new ApiError(detail, response.status);
  }
  return data;
}

export { ApiError };
