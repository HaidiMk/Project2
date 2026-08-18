import { ENDPOINTS } from "../config/api.js";

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
