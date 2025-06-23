import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create directories if they don't exist
const publicDir = path.join(__dirname, 'public');
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir);
}

// Try different possible worker file locations
const possibleWorkerPaths = [
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.js'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.mjs'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.min.js'),
];

let workerSrc = null;
for (const workerPath of possibleWorkerPaths) {
  if (fs.existsSync(workerPath)) {
    workerSrc = workerPath;
    break;
  }
}

if (!workerSrc) {
  console.error('Could not find PDF.js worker file in any of the expected locations');
  process.exit(1);
}

const workerDest = path.join(publicDir, 'pdf.worker.min.js');

try {
  fs.copyFileSync(workerSrc, workerDest);
  console.log('Successfully copied PDF.js worker file');
} catch (error) {
  console.error('Error copying PDF.js worker file:', error);
  process.exit(1);
}

// Copy cmaps
const cmapsDir = path.join(publicDir, 'cmaps');
if (!fs.existsSync(cmapsDir)) {
  fs.mkdirSync(cmapsDir, { recursive: true });
}

const cmapsSrc = path.join(__dirname, 'node_modules', 'pdfjs-dist', 'cmaps');
const cmapFiles = fs.readdirSync(cmapsSrc);

cmapFiles.forEach(file => {
  const srcPath = path.join(cmapsSrc, file);
  const destPath = path.join(cmapsDir, file);
  fs.copyFileSync(srcPath, destPath);
});
console.log('✅ Copied cmaps files');

// Copy standard fonts
const fontsDir = path.join(publicDir, 'standard_fonts');
if (!fs.existsSync(fontsDir)) {
  fs.mkdirSync(fontsDir, { recursive: true });
}

const fontsSrc = path.join(__dirname, 'node_modules', 'pdfjs-dist', 'standard_fonts');
const fontFiles = fs.readdirSync(fontsSrc);

fontFiles.forEach(file => {
  const srcPath = path.join(fontsSrc, file);
  const destPath = path.join(fontsDir, file);
  fs.copyFileSync(srcPath, destPath);
});
console.log('✅ Copied standard fonts files');

console.log('✅ PDF.js setup completed successfully'); 