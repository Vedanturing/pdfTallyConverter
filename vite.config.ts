import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';

// Copy PDF.js worker to public directory - prioritize react-pdf's internal pdfjs-dist
const possibleWorkerPaths = [
  path.resolve(__dirname, 'node_modules/react-pdf/node_modules/pdfjs-dist/build/pdf.worker.min.js'),
  path.resolve(__dirname, 'node_modules/pdfjs-dist/build/pdf.worker.min.js'),
];

const publicWorkerPath = path.resolve(__dirname, 'public/pdf.worker.min.js');

if (!fs.existsSync('public')) {
  fs.mkdirSync('public');
}

let workerFound = false;
for (const workerPath of possibleWorkerPaths) {
  if (fs.existsSync(workerPath)) {
    fs.copyFileSync(workerPath, publicWorkerPath);
    console.log(`Copied PDF.js worker from: ${workerPath}`);
    workerFound = true;
    break;
  }
}

if (!workerFound) {
  console.warn('Could not find PDF.js worker file');
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
          'react-pdf': ['react-pdf'],
        },
      },
    },
    commonjsOptions: {
      include: [/node_modules/],
    },
  },
  optimizeDeps: {
    include: ['react-pdf'],
  },
  worker: {
    format: 'es'
  },
  base: '/'
});

