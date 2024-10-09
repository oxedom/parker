import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

const fullReloadAlways = {
  name: 'full-reload',
  handleHotUpdate({ server }) {
    server.ws.send({ type: 'full-reload' })
    return []
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [fullReloadAlways, vue(), vueDevTools()],

  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
