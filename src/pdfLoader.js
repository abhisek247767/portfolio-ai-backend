const fs = require('fs');
const pdfParse = require('pdf-parse');

async function loadPdfText(pdfPath) {
  if (!fs.existsSync(pdfPath)) {
    throw new Error(`PDF not found at path: ${pdfPath}. Place your about-me PDF there and restart the server.`);
  }

  const buffer = fs.readFileSync(pdfPath);
  const data = await pdfParse(buffer);
  return data.text || '';
}

module.exports = {
  loadPdfText
};


