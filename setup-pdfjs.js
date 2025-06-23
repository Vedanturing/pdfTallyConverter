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

// Try different possible worker file locations - prioritize react-pdf's internal pdfjs-dist
const possibleWorkerPaths = [
  path.join(__dirname, 'node_modules', 'react-pdf', 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.mjs'),
  path.join(__dirname, 'node_modules', 'react-pdf', 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.mjs'),
  path.join(__dirname, 'node_modules', 'react-pdf', 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.js'),
  path.join(__dirname, 'node_modules', 'react-pdf', 'node_modules', 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.min.js'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.js'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.mjs'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.min.js'),
];

let workerSrc = null;
for (const workerPath of possibleWorkerPaths) {
  if (fs.existsSync(workerPath)) {
    workerSrc = workerPath;
    console.log(`Found PDF.js worker at: ${workerPath}`);
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

// Copy cmaps - prioritize react-pdf's internal cmaps
const cmapsDir = path.join(publicDir, 'cmaps');
if (!fs.existsSync(cmapsDir)) {
  fs.mkdirSync(cmapsDir, { recursive: true });
}

const possibleCmapsPaths = [
  path.join(__dirname, 'node_modules', 'react-pdf', 'node_modules', 'pdfjs-dist', 'cmaps'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'cmaps'),
];

let cmapsSrc = null;
for (const cmapsPath of possibleCmapsPaths) {
  if (fs.existsSync(cmapsPath)) {
    cmapsSrc = cmapsPath;
    console.log(`Found PDF.js cmaps at: ${cmapsPath}`);
    break;
  }
}

if (cmapsSrc) {
  try {
    const cmapFiles = fs.readdirSync(cmapsSrc);
    cmapFiles.forEach(file => {
      const srcPath = path.join(cmapsSrc, file);
      const destPath = path.join(cmapsDir, file);
      fs.copyFileSync(srcPath, destPath);
    });
    console.log('Successfully copied PDF.js cmaps');
  } catch (error) {
    console.error('Error copying PDF.js cmaps:', error);
  }
} else {
  console.warn('Could not find PDF.js cmaps directory');
}

// Copy standard fonts - prioritize react-pdf's internal fonts
const fontsDir = path.join(publicDir, 'standard_fonts');
if (!fs.existsSync(fontsDir)) {
  fs.mkdirSync(fontsDir, { recursive: true });
}

const possibleFontsPaths = [
  path.join(__dirname, 'node_modules', 'react-pdf', 'node_modules', 'pdfjs-dist', 'standard_fonts'),
  path.join(__dirname, 'node_modules', 'pdfjs-dist', 'standard_fonts'),
];

let fontsSrc = null;
for (const fontsPath of possibleFontsPaths) {
  if (fs.existsSync(fontsPath)) {
    fontsSrc = fontsPath;
    console.log(`Found PDF.js standard fonts at: ${fontsPath}`);
    break;
  }
}

if (fontsSrc) {
  try {
    const fontFiles = fs.readdirSync(fontsSrc);
    fontFiles.forEach(file => {
      const srcPath = path.join(fontsSrc, file);
      const destPath = path.join(fontsDir, file);
      fs.copyFileSync(srcPath, destPath);
    });
    console.log('Successfully copied PDF.js standard fonts');
  } catch (error) {
    console.error('Error copying PDF.js standard fonts:', error);
  }
} else {
  console.warn('Could not find PDF.js standard fonts directory');
}

console.log('✅ PDF.js setup completed successfully'); 