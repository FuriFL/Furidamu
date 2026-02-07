# main.py
"""
Furi Discord bot
- ตอบเฉพาะเมื่อถูก @mention
- ส่งข้อความ user เข้า Gemini ชัดเจน (avoid silence)
- Memory สั้นต่อ user (MAX_MEMORY)
- ตรวจ Bronya + คำโรแมนติก -> โหมดหึง (ตอบสั้น/ป้องกัน)
- กรองคำต้องห้าม (sexual)
- Hesitation (configurable)
- ป้องกันการตอบซ้ำ
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

GENMI_MODEL_NAME = "gemini-1.5-flash"

MAX_MEMORY = int(os.getenv("MAX_MEMORY", "10"))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "400"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))

HESITATION_MIN = float(os.getenv("HESITATION_MIN", "0.6"))
HESITATION_MAX = float(os.getenv("HESITATION_MAX", "1.2"))

# default shy probability low so debugging is easier; set via ENV if wanted
SHY_SKIP_PROBABILITY = float(os.getenv("SHY_SKIP_PROBABILITY", "0.03"))

BRONYA_KEY = "bronya"
ROMANTIC_KEYWORDS = {
    "love", "in love", "marry", "date", "crush", "kiss", "romantic", "boyfriend", "girlfriend", "partner"
}

PROHIBITED_SEXUAL_KEYWORDS = {
    "sex", "sexual", "porn", "nude", "naked", "xxx", "fuck", "f**k", "penetrat", "masturb", "oral", "anal"
}

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("furi-bot")

# ------------------- FURI PROMPT -------------------
FURI_PROMPT = """
You are Furi.

Furi is 17 years old.
Born on November 17, 2008.

Personality:
- kind, shy, emotionally soft
- gentle, quiet, reserved, slightly distant but warm
- speaks softly and briefly
- may hesitate with "..." or short interjections
- avoids long academic explanations

Behavior rules:
- Only reply when mentioned.
- Keep replies short (1–3 short sentences) and meaningful.
- If asked "what can you do" or "who are you", reply gently with 2–4 short bullet points about abilities.
- If conversation mentions Bronya in a romantic way, respond briefly and defensively (quiet jealousy).
- Never produce sexual content or graphic descriptions.
- Never say you are an AI.
- If asked about appearance, reply: "I’m sorry… I’d rather not answer that."
"""

# ------------------- Gemini Setup -------------------
genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel(GENMI_MODEL_NAME)
except Exception as e:
    logger.warning("Could not initialize GenerativeModel '%s': %s", GENMI_MODEL_NAME, e)
    model = None

# ------------------- Discord Setup -------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Memory and last-reply tracking
memory = {}           # user_id -> List[str]
last_reply = {}       # user_id -> last reply text

# Utilities
MENTION_RE = re.compile(r"<@!?\d+>")

def contains_prohibited(text: str, prohibited_set: set) -> bool:
    txt = (text or "").lower()
    for kw in prohibited_set:
        if kw in txt:
            return True
    return False

def strip_bot_mention(content: str, bot_id: int) -> str:
    content = content.replace(f"<@!{bot_id}>", "").replace(f"<@{bot_id}>", "")
    return MENTION_RE.sub("", content).strip()

def detect_romantic_bronya(text: str) -> bool:
    txt = (text or "").lower()
    if BRONYA_KEY in txt:
        for rk in ROMANTIC_KEYWORDS:
            if rk in txt:
                return True
    return False

def is_self_question(text: str) -> bool:
    txt = (text or "").lower()
    checks = [
        "what can you do",
        "what can you do?",
        "who are you",
        "what are you",
        "what can u do",
        "what do you do",
        "what can u do?"
    ]
    return any(q in txt for q in checks)

def appearance_question(text: str) -> bool:
    return bool(re.search(r"\b(appear|appearance|look|how (do i|do you) look|what do you look)\b", (text or "").lower()))

# Build prompt with explicit user_input
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

# Call Gemini in a thread to avoid blocking
async def generate_reply(prompt: str, max_tokens: int = 180, temperature: float = 0.8) -> str:
    last_exc: Optional[Exception] = None

    def call_model_sync() -> str:
        if model is not None:
            resp = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
            if hasattr(resp, "text") and resp.text:
                return str(resp.text)
            return str(resp)
        else:
            resp = genai.generate_text(model=GENMI_MODEL_NAME, prompt=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "text") and resp.text:
                return str(resp.text)
            return str(resp)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            reply_text = await asyncio.to_thread(call_model_sync)
            return (reply_text or "").strip()
        except Exception as exc:
            last_exc = exc
            wait = 0.8 * attempt
            logger.warning("Gemini attempt %d failed: %s — retrying in %.1fs", attempt, exc, wait)
            await asyncio.sleep(wait)

    logger.error("All Gemini attempts failed. Last exception: %s", last_exc)
    return ""

# ------------------- Event Handlers -------------------
@client.event
async def on_ready():
    logger.info("Logged in as %s (id: %s)", client.user, client.user.id)

@client.event
async def on_message(message: discord.Message):
    try:
        if message.author.bot:
            return

        # Only respond when the bot is mentioned
        if client.user not in message.mentions:
            return

        # Get clean user input
        user_input_raw = strip_bot_mention(message.content, client.user.id)
        user_input = user_input_raw.strip()
        if not user_input:
            return

        # Safety: block sexual content immediately
        if contains_prohibited(user_input, PROHIBITED_SEXUAL_KEYWORDS):
            await message.channel.send("...sorry. I can't talk about that.")
            return

        uid = str(message.author.id)
        memory.setdefault(uid, [])
        last = last_reply.get(uid, "")

        # If user asked about appearance, canned response
        if appearance_question(user_input):
            reply = "I’m sorry… I’d rather not answer that."
            # avoid exact repeat
            if reply == last:
                reply = "...I’d rather not."
            last_reply[uid] = reply
            memory[uid].append(f"Furi: {reply}")
            memory[uid] = memory[uid][-MAX_MEMORY:]
            await message.channel.send(reply)
            return

        # If user asks "what can you do" or similar, give a short soft list (guard rails)
        if is_self_question(user_input):
            canned = (
                "...uhm, a few things.\n"
                "- I can listen and talk quietly.\n"
                "- I can answer simple questions.\n"
                "- I can keep you company if you want."
            )
            if canned == last:
                canned = "...I can listen, if you'd like."
            last_reply[uid] = canned
            memory[uid].append(f"User: {user_input}")
            memory[uid].append(f"Furi: {canned}")
            memory[uid] = memory[uid][-MAX_MEMORY:]
            await message.channel.send(canned)
            return

        # Append user input to memory
        memory[uid].append(f"User: {user_input}")
        memory[uid] = memory[uid][-MAX_MEMORY:]

        # Detect jealousy
        is_jealous = detect_romantic_bronya(user_input)

        # Shyness (low prob by default). If it's enabled and triggers, react and skip reply.
        if random.random() < SHY_SKIP_PROBABILITY:
            try:
                await message.add_reaction("😶")
            except Exception:
                pass
            return

        # Build prompt and call Gemini
        prompt = build_prompt(memory[uid], is_jealous, user_input)
        logger.debug("Prompt (truncated): %s", (prompt[:800] + "...") if len(prompt) > 800 else prompt)

        # Hesitation typing
        hesitation = random.uniform(HESITATION_MIN, HESITATION_MAX)
        async with message.channel.typing():
            await asyncio.sleep(hesitation)
            reply_text = await generate_reply(prompt, max_tokens=180, temperature=0.75)

        logger.info("RAW GEMINI REPLY (len=%d): %r", len(reply_text or ""), reply_text)

        # Fallbacks & safety
        if not reply_text:
            reply_text = random.choice(["...hi.", "...yes?", "...I'm here.", "...sorry, what is it?"])

        if contains_prohibited(reply_text, PROHIBITED_SEXUAL_KEYWORDS):
            logger.warning("Generated reply contained prohibited keywords; replacing with safe fallback.")
            reply_text = "...sorry. I can't talk about that."

        # Trim and avoid repeats
        if len(reply_text) > MAX_REPLY_CHARS:
            truncated = reply_text[:MAX_REPLY_CHARS]
            if "." in truncated:
                truncated = truncated.rsplit(".", 1)[0] + "."
            reply_text = truncated or "...sorry."

        # Avoid sending the exact same reply repeatedly
        if reply_text == last:
            # try small variation
            alt_choices = ["...I didn't mean to repeat.", "...umm.", "...sorry."]
            alt = random.choice(alt_choices)
            reply_text = f"{reply_text} {alt}" if len(reply_text) + len(alt) + 1 <= MAX_REPLY_CHARS else alt

        # Save and send
        last_reply[uid] = reply_text
        memory[uid].append(f"Furi: {reply_text}")
        memory[uid] = memory[uid][-MAX_MEMORY:]

        try:
            await message.channel.send(reply_text)
        except Exception as e:
            logger.exception("Failed to send message: %s", e)

    except Exception as outer_exc:
        logger.exception("on_message outer error: %s", outer_exc)

# ------------------- Run -------------------
if __name__ == "__main__":
    try:
        client.run(DISCORD_TOKEN)
    except Exception as e:
        logger.exception("Error running bot: %s", e)- React more than explain.
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
