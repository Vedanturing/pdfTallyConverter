import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';

// Copy PDF.js worker to public directory
const workerPath = path.resolve(__dirname, 'node_modules/pdfjs-dist/build/pdf.worker.min.js');
const publicWorkerPath = path.resolve(__dirname, 'public/pdf.worker.min.js');

if (!fs.existsSync('public')) {
  fs.mkdirSync('public');
}

if (fs.existsSync(workerPath)) {
  fs.copyFileSync(workerPath, publicWorkerPath);
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    host: true,
    cors: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        secure: false
      },
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/convert': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/file': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  css: {
    postcss: './postcss.config.cjs',
    devSourcemap: true
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html')
      },
      output: {
        manualChunks: {
          pdfjs: ['pdfjs-dist'],
          'react-pdf': ['react-pdf'],
        },
      },
    },
    commonjsOptions: {
      include: [/node_modules/],
    },
  },
  optimizeDeps: {
    include: ['react-pdf', 'pdfjs-dist'],
  },
  worker: {
    format: 'es'
  },
  base: '/'
});

