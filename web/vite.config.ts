import viteReact from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import viteSvgr from 'vite-plugin-svgr';
import viteTsconfigPaths from 'vite-tsconfig-paths';

// https://vitejs.dev/config/
export default defineConfig({
  base: '/',
  plugins: [viteReact(), viteTsconfigPaths(), viteSvgr()],
  server: {
    host: true,
    port: 3030,
    proxy: {
      '/api': {
        target: 'http://localhost:5000', // Replace with your back-end server URL
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
