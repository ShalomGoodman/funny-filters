import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
  },
  preview: {
    host: '0.0.0.0',
    strictPort: false,
    allowedHosts: [
      '.herokuapp.com',
      'funny-filters-68352a4b37a8.herokuapp.com'
    ]
  }
})
