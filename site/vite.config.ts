import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1];
const base = process.env.GITHUB_ACTIONS && repository ? `/${repository}/` : '/';

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  server: {
    fs: {
      allow: [path.resolve(__dirname, '..')]
    }
  }
});

