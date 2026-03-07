const path = require('path');
const fs = require('fs');
const express = require('express');
const cors = require('cors');
require('dotenv').config();

const { initializeRag, getAnswerForQuery, isInitialized } = require('./src/rag');

const app = express();
const PORT = process.env.PORT || 4000;

const allowedOrigin = process.env.FRONTEND_ORIGIN;

app.use(
  cors({
    origin: allowedOrigin || '*'
  })
);
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    ragInitialized: isInitialized()
  });
});

app.post('/chat', async (req, res) => {
  try {
    const { message, history } = req.body || {};

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }

    const safeHistory = Array.isArray(history) ? history.slice(-10) : [];

    const answer = await getAnswerForQuery(message, safeHistory);

    res.json({ reply: answer });
  } catch (err) {
    console.error('Error in /chat:', err);
    res.status(500).json({ error: 'Something went wrong while generating a response.' });
  }
});

async function start() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.warn('Warning: GEMINI_API_KEY is not set. The RAG chatbot will not work until you configure it in a .env file.');
  }

  const pdfPath = path.join(__dirname, 'data', 'abhisek_profile.pdf');
  try {
    await initializeRag(pdfPath);
    console.log('RAG initialization complete.');
  } catch (err) {
    console.error('Failed to initialize RAG from PDF:', err.message || err);
  }

  app.listen(PORT, () => {
    console.log(`RAG backend listening on port ${PORT}`);
  });
}

start().catch(err => {
  console.error('Fatal error while starting server:', err);
  process.exit(1);
});


