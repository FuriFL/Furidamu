"""
Discord bot "Furi"
- ตอบเฉพาะเมื่อถูก @mention เท่านั้น
- ใช้ Gemini via google-generativeai
- เก็บ memory สั้นต่อ user (MAX_MEMORY)
- มี logic ตรวจคำพูดเกี่ยวกับ "Bronya" และคำโรแมนติก -> โหมดหึง (ตอบสั้น/ป้องกัน)
- ป้องกันข้อความที่เป็นเนื้อหาไม่เหมาะสม (sexual)
- ใส่ hesitation เพื่อให้รู้สึกเป็นคนจริง ๆ
"""

import os
import re
import asyncio
import random
import logging
from typing import List

import discord
import google.generativeai as genai

# ------------------- Configuration -------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Error: DISCORD_TOKEN and GEMINI_API_KEY must be set in environment variables.")

# Gemini model to use (change if you have a different preferred model)
GENMI_MODEL_NAME = "gemini-1.5-flash"

# Memory settings
MAX_MEMORY = 10  # keep last N messages per user

# Response limits
MAX_REPLY_CHARS = 400  # ensure responses are short
RETRY_ATTEMPTS = 3

# Hesitation parameters (seconds)
HESITATION_MIN = 0.8
HESITATION_MAX = 1.6

# Probability that Furi is "too shy to answer" even when mentioned (0-1)
SHY_SKIP_PROBABILITY = 0.12

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

# ------------------- FURI PROMPT -------------------
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
- short replies (1–3 sentences)
- may use "..." or quiet reactions
- not overly expressive
- not energetic or loud
- emotionally aware but restrained

Self-awareness:
Furi knows clearly that her name is Furi and who she is.
She does not question her identity.

Appearance:
Furi never talks about her appearance unless directly asked.
If asked about her appearance, she replies:
"I’m sorry… I’d rather not answer that."

Interests and boundaries:
Furi loves Bronya deeply.
She becomes quiet, defensive, or jealous if someone talks about Bronya romantically.
She does not flirt with others.
Furi identifies as lesbian and is not interested in men in any way.

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
    # Fall back to a generic interface if needed; still keep genai configured.
    model = None

# ------------------- Discord Setup -------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Memory store: user_id (str) -> List[str]
memory = {}

# Utility: sanitize and check for prohibited content
def contains_prohibited(text: str, prohibited_set: set) -> bool:
    txt = text.lower()
    for kw in prohibited_set:
        if kw in txt:
            return True
    return False

# Utility: clean mention tokens like <@1234567890> or <@!1234567890>
MENTION_RE = re.compile(r"<@!?\d+>")

def strip_mentions(text: str) -> str:
    # Remove mention tokens
    return MENTION_RE.sub("", text).strip()

def detect_romantic_bronya(text: str) -> bool:
    txt = text.lower()
    if BRONYA_KEY in txt:
        for rk in ROMANTIC_KEYWORDS:
            if rk in txt:
                return True
    return False

# Build the prompt we send to Gemini
def build_prompt(chat_history: List[str], is_jealous: bool) -> str:
    """
    chat_history: list of lines like "User: ...", "Furi: ..."
    is_jealous: if True, instruct model to be defensive/jealous about Bronya
    """
    extra = ""
    if is_jealous:
        extra = (
            "\n\nNOTE: This conversation includes romantic talk about Bronya. "
            "Furi should respond briefly, defensively, and with quiet jealousy. "
            "Keep replies short (1 sentence), restrained, and avoid romanticizing."
        )
    # Safety guard reminder in prompt: no sexual content, no explicit romance
    safety = (
        "\n\nIMPORTANT: Do not produce sexual content, explicit descriptions, or simulate sexual acts. "
        "Keep everything age-appropriate and non-sexual."
    )
    history = "\n".join(chat_history[-MAX_MEMORY:])
    prompt = f"{FURI_PROMPT}\n\nConversation:\n{history}\n\nFuri's reply:{extra}{safety}\n"
    return prompt

# Async function to call Gemini with retry/backoff
async def generate_reply(prompt: str, max_tokens: int = 180, temperature: float = 0.8) -> str:
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            if model is not None:
                # Using the simple generate_content interface if available
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
                # Some SDKs return text in different attributes; using .text if present
                reply_text = getattr(response, "text", None) or str(response)
            else:
                # If model object wasn't created, fallback to genai.chat.create style call
                response = genai.generate_text(model=GENMI_MODEL_NAME, prompt=prompt, max_output_tokens=max_tokens)
                reply_text = response.text if hasattr(response, "text") else str(response)
            # Ensure string
            return str(reply_text).strip()
        except Exception as exc:
            last_exc = exc
            wait = 1.0 * attempt
            logger.warning("Gemini generate attempt %s failed: %s — retrying in %.1fs", attempt, exc, wait)
            await asyncio.sleep(wait)
    logger.error("All Gemini attempts failed. Last exception: %s", last_exc)
    return ""  # empty signals upstream to use fallback reply

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

    # Remove mentions from the content to get user input
    user_input = strip_mentions(message.content)
    if not user_input:
        # If someone only mentioned without text, optionally ignore
        return

    # Basic content safety: if the user input itself contains sexual content, do not forward to model
    if contains_prohibited(user_input, PROHIBITED_SEXUAL_KEYWORDS):
        await message.channel.send("...sorry. I can't talk about that.")
        return

    user_id = str(message.author.id)
    if user_id not in memory:
        memory[user_id] = []

    # Append user message to memory
    memory[user_id].append(f"User: {user_input}")
    memory[user_id] = memory[user_id][-MAX_MEMORY:]

    # Determine if this mention is about Bronya romantically
    is_jealous = detect_romantic_bronya(user_input)

    # Shyness: sometimes Furi chooses not to reply (small chance)
    if random.random() < SHY_SKIP_PROBABILITY:
        # Option: send a very short hesitant reaction or send nothing.
        # We'll send a very quiet ellipsis to indicate shyness.
        try:
            await message.add_reaction("😶")
        except Exception:
            pass
        return

    # Build prompt including memory
    prompt = build_prompt(memory[user_id], is_jealous)

    # Simulate thinking/hesitation
    hesitation = random.uniform(HESITATION_MIN, HESITATION_MAX)
    async with message.channel.typing():
        await asyncio.sleep(hesitation)

        # Call Gemini
        reply_text = await generate_reply(prompt, max_tokens=180, temperature=0.75)

    # Fallback if model failed
    if not reply_text:
        reply_text = "...sorry. I feel a bit quiet right now."

    # Post-check: ensure reply does not contain any prohibited terms
    if contains_prohibited(reply_text, PROHIBITED_SEXUAL_KEYWORDS):
        logger.warning("Generated reply contained prohibited keywords; replacing with safe fallback.")
        reply_text = "...sorry. I can't talk about that."

    # Truncate to limit
    if len(reply_text) > MAX_REPLY_CHARS:
        reply_text = reply_text[:MAX_REPLY_CHARS].rsplit(".", 1)[0]  # try to cut at sentence end
        reply_text = (reply_text + "...") if reply_text else "...sorry."

    # Additional safety: If the user asked about appearance explicitly, force the canned response:
    if re.search(r"\b(appear|appearance|look|how (do i|do you) look|what do you look)\b", user_input.lower()):
        reply_text = "I’m sorry… I’d rather not answer that."

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
