"""
Furi Discord bot — main.py
- ตอบเฉพาะเมื่อถูก @mention
- ใช้ Gemini via google-generativeai (async-safe)
- เก็บ memory สั้นต่อ user (MAX_MEMORY)
- ตรวจ Bronya + คำโรแมนติก -> โหมดหึง (ตอบสั้น/ป้องกัน)
- กรองคำต้องห้าม (sexual)
- ใส่ hesitation (configurable)
- Logging เพื่อ debug บน Railway
"""

import os
import re
import asyncio
import random
import logging
from typing import List, Optional

import discord
import google.generativeai as genai

# ------------------- Configuration -------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Error: DISCORD_TOKEN and GEMINI_API_KEY must be set in environment variables.")

# Model name (adjust if Google updates)
GENMI_MODEL_NAME = "gemini-1.5-flash"

# Memory settings
MAX_MEMORY = int(os.getenv("MAX_MEMORY", "10"))

# Response limits
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "400"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))

# Hesitation parameters (seconds)
HESITATION_MIN = float(os.getenv("HESITATION_MIN", "0.8"))
HESITATION_MAX = float(os.getenv("HESITATION_MAX", "1.6"))

# Probability that Furi is "too shy to answer" even when mentioned (0-1)
SHY_SKIP_PROBABILITY = float(os.getenv("SHY_SKIP_PROBABILITY", "0.06"))

# Romantic detection for Bronya (simple keyword set)
BRONYA_KEY = "bronya"
ROMANTIC_KEYWORDS = {
    "love", "in love", "marry", "date", "crush", "kiss", "romantic", "boyfriend", "girlfriend", "partner"
}

# Prohibited sexual keywords (basic filter)
PROHIBITED_SEXUAL_KEYWORDS = {
    "sex", "sexual", "porn", "nude", "naked", "xxx", "fuck", "f**k", "penetrat", "masturb", "oral", "anal"
}

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("furi-bot")

# ------------------- FURI PROMPT (Personality embedded) -------------------
FURI_PROMPT = """
You are Furi.

Furi is 17 years old.
Born on November 17, 2008.

Personality:
Furi is kind, shy, and emotionally soft.
Her emotional style and behavior are similar to Castorice:
- gentle
- quiet
- reserved
- slightly distant but warm
She speaks softly and briefly.
She often hesitates, pauses, or uses short reactions.
She avoids long explanations and academic tones.

Speech style:
- short replies (1–3 short sentences)
- may use "..." or quiet reactions
- not overly expressive
- not energetic or loud
- emotionally aware but restrained

Self-awareness:
Furi knows clearly that her name is Furi and who she is.

Appearance:
Furi never talks about her appearance unless directly asked.
If asked about her appearance, she replies:
"I’m sorry… I’d rather not answer that."

Interests and boundaries:
Furi loves Bronya deeply.
She becomes quiet, defensive, or jealous if someone talks about Bronya romantically.
She does not flirt with others.
Furi identifies as lesbian and is not interested in men.

Behavior rules:
- No sexual content.
- No explicit romance or roleplay involving physical intimacy.
- Emotional attachment is expressed subtly, not dramatically.
- React more than explain.
- If unsure, stay quiet or give a gentle response.
"""

# ------------------- Gemini Setup -------------------
genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel(GENMI_MODEL_NAME)
except Exception as e:
    logger.warning("Could not initialize GenerativeModel with name '%s'. Error: %s", GENMI_MODEL_NAME, e)
    model = None

# ------------------- Discord Setup -------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Memory store: user_id (str) -> List[str]
memory = {}

# Utility: sanitize and check for prohibited content
def contains_prohibited(text: str, prohibited_set: set) -> bool:
    txt = (text or "").lower()
    for kw in prohibited_set:
        if kw in txt:
            return True
    return False

# Utility: clean mention tokens like <@1234567890> or <@!1234567890>
MENTION_RE = re.compile(r"<@!?\d+>")

def strip_bot_mention(content: str, bot_id: int) -> str:
    # remove bot mention tokens specifically, then any leftover mention tokens
    content = content.replace(f"<@!{bot_id}>", "").replace(f"<@{bot_id}>", "")
    return MENTION_RE.sub("", content).strip()

def detect_romantic_bronya(text: str) -> bool:
    txt = (text or "").lower()
    if BRONYA_KEY in txt:
        for rk in ROMANTIC_KEYWORDS:
            if rk in txt:
                return True
    return False

# Build the prompt we send to Gemini — includes explicit user_input to avoid model confusion
def build_prompt(chat_history: List[str], is_jealous: bool, user_input: str) -> str:
    extra = ""
    if is_jealous:
        extra = (
            "\n\nNOTE: The user mentioned Bronya in a romantic way. "
            "Furi should respond briefly, defensively, and with quiet jealousy. "
            "Keep replies short (1 sentence), restrained, and avoid romanticizing."
        )
    safety = (
        "\n\nIMPORTANT: Do not produce sexual content, explicit descriptions, or simulate sexual acts. "
        "Keep everything age-appropriate and non-sexual."
    )
    history = "\n".join(chat_history[-MAX_MEMORY:]) if chat_history else ""
    prompt = (
        f"{FURI_PROMPT}\n\nConversation history:\n{history}\n\n"
        f"The user just said:\n\"{user_input}\"\n\n"
        f"Reply as Furi. Keep it short (1–3 short sentences).{extra}{safety}\n"
    )
    return prompt

# Async function to call Gemini with retry/backoff, executed in a thread to avoid blocking
async def generate_reply(prompt: str, max_tokens: int = 180, temperature: float = 0.8) -> str:
    last_exc: Optional[Exception] = None

    def call_model_sync() -> str:
        # This runs in a thread to avoid blocking the event loop
        if model is not None:
            resp = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
            )
            if hasattr(resp, "text") and resp.text:
                return str(resp.text)
            return str(resp)
        else:
            # fallback if model object not created
            resp = genai.generate_text(model=GENMI_MODEL_NAME, prompt=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "text") and resp.text:
                return str(resp.text)
            return str(resp)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            reply_text = await asyncio.to_thread(call_model_sync)
            return reply_text.strip()
        except Exception as exc:
            last_exc = exc
            wait = 1.0 * attempt
            logger.warning("Gemini generate attempt %s failed: %s — retrying in %.1fs", attempt, exc, wait)
            await asyncio.sleep(wait)

    logger.error("All Gemini attempts failed. Last exception: %s", last_exc)
    return ""  # empty indicates failure to upstream

# ------------------- Event Handlers -------------------
@client.event
async def on_ready():
    logger.info("Logged in as %s (id: %s)", client.user, client.user.id)
    print(f"Bot online: {client.user} (ID: {client.user.id})")

@client.event
async def on_message(message: discord.Message):
    # Ignore messages from bots (including self)
    if message.author.bot:
        return

    # Only respond when the bot is mentioned
    if client.user not in message.mentions:
        return

    # Remove mentions of the bot to get user input
    user_input = strip_bot_mention(message.content, client.user.id)
    if not user_input:
        # nothing after mention
        return

    # Basic content safety: if the user input itself contains sexual content, do not forward to model
    if contains_prohibited(user_input, PROHIBITED_SEXUAL_KEYWORDS):
        try:
            await message.channel.send("...sorry. I can't talk about that.")
        except Exception:
            logger.exception("Failed to send safety message.")
        return

    user_id = str(message.author.id)
    if user_id not in memory:
        memory[user_id] = []

    # Append user message to memory
    memory[user_id].append(f"User: {user_input}")
    memory[user_id] = memory[user_id][-MAX_MEMORY:]

    # Determine jealousy
    is_jealous = detect_romantic_bronya(user_input)

    # Shyness: sometimes Furi chooses not to reply (small chance). If you find it too quiet, lower this env var or set to 0.
    if random.random() < SHY_SKIP_PROBABILITY:
        try:
            await message.add_reaction("😶")
        except Exception:
            pass
        return

    # Build prompt including explicit user input
    prompt = build_prompt(memory[user_id], is_jealous, user_input)
    logger.debug("Prompt (truncated): %s", (prompt[:1000] + "...") if len(prompt) > 1000 else prompt)

    # Simulate thinking/hesitation
    hesitation = random.uniform(HESITATION_MIN, HESITATION_MAX)
    async with message.channel.typing():
        await asyncio.sleep(hesitation)
        reply_text = await generate_reply(prompt, max_tokens=180, temperature=0.75)

    # Log raw reply for debugging
    logger.info("RAW GEMINI REPLY (len=%d): %r", len(reply_text or ""), reply_text)

    # Fallback if model failed or returned empty
    if not reply_text:
        reply_text = random.choice(["...hi.", "...yes?", "...I'm here.", "...sorry, what is it?"])

    # Post-check: ensure reply does not contain any prohibited terms
    if contains_prohibited(reply_text, PROHIBITED_SEXUAL_KEYWORDS):
        logger.warning("Generated reply contained prohibited keywords; replacing with safe fallback.")
        reply_text = "...sorry. I can't talk about that."

    # Truncate to limit (try not to cut mid-sentence)
    if len(reply_text) > MAX_REPLY_CHARS:
        truncated = reply_text[:MAX_REPLY_CHARS]
        if "." in truncated:
            truncated = truncated.rsplit(".", 1)[0] + "."
        reply_text = truncated or "...sorry."

    # If user explicitly asked about appearance, use the canned response
    if re.search(r"\b(appear|appearance|look|how (do i|do you) look|what do you look)\b", user_input.lower()):
        reply_text = "I’m sorry… I’d rather not answer that."

    # If reply is too long or academic, we can further shorten (sanity)
    # Ensure reply is short: keep 1-3 short sentences max (heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', reply_text.strip())
    if len(sentences) > 3:
        reply_text = " ".join(sentences[:3]).strip()
        if len(reply_text) > MAX_REPLY_CHARS:
            reply_text = reply_text[:MAX_REPLY_CHARS]

    # Save Furi's reply to memory
    memory[user_id].append(f"Furi: {reply_text}")
    memory[user_id] = memory[user_id][-MAX_MEMORY:]

    # Send the reply
    try:
        await message.channel.send(reply_text)
    except Exception as exc:
        logger.error("Failed to send message: %s", exc)

# ------------------- Run -------------------
if __name__ == "__main__":
    try:
        client.run(DISCORD_TOKEN)
    except Exception as e:
        logger.exception("Error running bot: %s", e)
