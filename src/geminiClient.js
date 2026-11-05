require("dotenv").config();
const { GoogleGenAI } = require("@google/genai");
const { systemPrompt, generationConfig } = require("./config/generationConfig");

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_CHAT_MODEL = process.env.GEMINI_CHAT_MODEL;
const GEMINI_EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL;

if (!GEMINI_API_KEY || !GEMINI_CHAT_MODEL || !GEMINI_EMBEDDING_MODEL) {
  console.warn("One or more required GEMINI environment variables are not set.");
}

const ai = new GoogleGenAI({
  apiKey: GEMINI_API_KEY,
});


const CHAT_MAX_RETRIES = parseInt(process.env.GEMINI_CHAT_MAX_RETRIES, 10);
const CHAT_RETRY_DELAY_MS = parseInt(process.env.GEMINI_CHAT_RETRY_DELAY_MS, 10);

if (Number.isNaN(CHAT_MAX_RETRIES) || Number.isNaN(CHAT_RETRY_DELAY_MS)) {
  console.warn("GEMINI_CHAT_MAX_RETRIES or GEMINI_CHAT_RETRY_DELAY_MS is not set; chat calls will not be retried.");
}

// 429 = rate limited / quota, 500/503/504 = model overloaded or temporarily down.
// Other errors (400 bad request, 403 key, 404 model) won't fix themselves, so no retry.
const RETRYABLE_STATUS = new Set([429, 500, 503, 504]);

function isRetryable(error) {
  return RETRYABLE_STATUS.has(error?.status);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function withRetry(fn, label) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (!(attempt < CHAT_MAX_RETRIES) || !isRetryable(error)) throw error;
      const delay = CHAT_RETRY_DELAY_MS * 2 ** attempt + Math.floor(Math.random() * 250);
      console.warn(
        `${label} failed with status ${error.status}; retry ${attempt + 1}/${CHAT_MAX_RETRIES} in ${delay}ms`
      );
      await sleep(delay);
    }
  }
}

async function generateChatCompletion(userMessage) {
  try {
    const response = await withRetry(
      () =>
        ai.models.generateContent({
          model: GEMINI_CHAT_MODEL,

          contents: [
            {
              role: "user",
              parts: [
                {
                  text: `${systemPrompt}\n\nUser Question: ${userMessage}`,
                },
              ],
            },
          ],

          generationConfig,
        }),
      "Chat completion"
    );

    return response.text;
  } catch (error) {
    console.error("Error generating chat completion:", error);
    return `- Apologies, something went wrong.
- Please try again in a moment.
- If the issue persists, feel free to contact Abhisek directly.`;
  }
}

module.exports = {
  generateChatCompletion,
};


