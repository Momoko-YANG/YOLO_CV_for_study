import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI during dev — avoids CORS issues
      "/auth": "http://localhost:8000",
      "/detect": "http://localhost:8000",
    },
  },
});
