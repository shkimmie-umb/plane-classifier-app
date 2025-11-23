import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // REPLACE 'repo-name' WITH YOUR ACTUAL GITHUB REPOSITORY NAME
  // Example: If your URL is https://shkimmie-umb.github.io/mri-app/
  // Then use base: '/mri-app/'
  base: '/plane-classifier-app/', 
})