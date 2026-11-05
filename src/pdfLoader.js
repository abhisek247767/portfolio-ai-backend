const fs = require('fs');
const pdfParse = require('pdf-parse');

// A text item lies inside a link if its baseline start point is within the
// link's rectangle (PDF coordinates: [x1, y1, x2, y2]).
function isInside(item, rect) {
  const [x1, y1, x2, y2] = rect;
  const x = item.transform[4];
  const y = item.transform[5];
  const pad = 2;
  return (
    x >= Math.min(x1, x2) - pad && x <= Math.max(x1, x2) + pad &&
    y >= Math.min(y1, y2) - pad && y <= Math.max(y1, y2) + pad
  );
}

// Resume PDFs often draw icons with an icon font; those come through as stray
// single characters or private-use glyphs. Drop them from link labels.
function cleanAnchor(text) {
  return text
    .replace(/[-]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 1)
    .join(' ');
}

const SITE_NAMES = [
  [/(^|\.)github\.com$/i, 'GitHub'],
  [/\.github\.io$/i, 'Portfolio website'],
  [/(^|\.)linkedin\.com$/i, 'LinkedIn'],
  [/\.vercel\.app$/i, 'Live demo']
];

function siteName(url) {
  try {
    const host = new URL(url).hostname;
    const hit = SITE_NAMES.find(([re]) => re.test(host));
    return hit ? hit[1] : host.replace(/^www\./, '');
  } catch {
    return '';
  }
}

/**
 * Reads the visible text of a PDF plus the URLs behind its hyperlinks.
 * pdf-parse alone returns only visible text, so a link shown as "AbhisekRoy"
 * would lose its real target (e.g. https://github.com/...). The links are
 * appended as a "Links" section so they get embedded and the bot can cite them.
 */
async function loadPdfText(pdfPath) {
  if (!fs.existsSync(pdfPath)) {
    throw new Error(`PDF not found at path: ${pdfPath}. Place your about-me PDF there and restart the server.`);
  }

  const buffer = fs.readFileSync(pdfPath);
  const links = new Map(); // url -> anchor text

  // Same text output as pdf-parse's default renderer, plus link collection.
  async function renderPage(page) {
    const content = await page.getTextContent({
      normalizeWhitespace: false,
      disableCombineTextItems: false
    });

    let lastY;
    let text = '';
    for (const item of content.items) {
      text += lastY === item.transform[5] || lastY === undefined ? item.str : `\n${item.str}`;
      lastY = item.transform[5];
    }

    for (const a of await page.getAnnotations()) {
      if (a.subtype !== 'Link' || !/^https?:\/\//i.test(a.url || '')) continue;
      const anchor = cleanAnchor(
        content.items
          .filter((item) => item.str.trim() && isInside(item, a.rect))
          .map((item) => item.str.trim())
          .join(' ')
      );
      if (!links.has(a.url) || (!links.get(a.url) && anchor)) links.set(a.url, anchor);
    }

    return text;
  }

  const data = await pdfParse(buffer, { pagerender: renderPage });
  let text = data.text || '';

  if (links.size > 0) {
    // e.g. "- GitHub (AbhisekRoy): https://github.com/abhisek247767"
    const lines = [...links].map(([url, anchor]) => {
      const site = siteName(url);
      const label = anchor && anchor !== site ? `${site} (${anchor})` : site;
      return `- ${label}: ${url}`;
    });
    text += `\n\nLinks:\n${lines.join('\n')}\n`;
  }

  return text;
}

module.exports = {
  loadPdfText
};
