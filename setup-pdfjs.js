import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create necessary directories
const publicDir = path.join(__dirname, 'public');
const cmapsDir = path.join(publicDir, 'cmaps');
const standardFontsDir = path.join(publicDir, 'standard_fonts');

// Create directories if they don't exist
[cmapsDir, standardFontsDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Copy files from node_modules
const pdfJsPath = path.join(__dirname, 'node_modules', 'pdfjs-dist');
const cmapsSource = path.join(pdfJsPath, 'cmaps');
const standardFontsSource = path.join(pdfJsPath, 'standard_fonts');

try {
  // Copy cmaps
  execSync(`xcopy "${cmapsSource}" "${cmapsDir}" /E /I /Y`, { stdio: 'inherit' });
  console.log('✅ Copied cmaps files');

  // Copy standard fonts
  execSync(`xcopy "${standardFontsSource}" "${standardFontsDir}" /E /I /Y`, { stdio: 'inherit' });
  console.log('✅ Copied standard fonts files');

  console.log('✅ PDF.js setup completed successfully');
} catch (error) {
  console.error('❌ Error during setup:', error);
  process.exit(1);
} 