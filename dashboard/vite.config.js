import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standalone Vite config for the Dashboard project only.
// This file has no relationship to the Django backend or any other
// part of the repo - it only configures how THIS React app is built/served.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
