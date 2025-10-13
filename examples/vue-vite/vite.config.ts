import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  optimizeDeps: {
    exclude: ["@jayce789/numjs"],
  },
  server: {
    port: 4173,
  },
});
