# main.py
# Furi Discord bot — Python (discord.py)
# - ตอบเฉพาะเมื่อถูก @mention
# - ลบ mention แบบปลอดภัย (รองรับ <@ID> และ <@!ID>)
# - ส่งข้อความ user ให้ Gemini ชัดเจน
# - memory สั้น ต่อ user
# - หึงเมื่อพูดถึง Bronya แบบโรแมนติก
# - กรองคำไม่เหมาะสม (sexual)
# - ไม่มี triple-quoted string literals ที่เสี่ยงเกิด SyntaxError

import os
import re
import asyncio
import random
import logging
from typing import List, Optional

import discord
import google.generativeai as genai
from dotenv import load_dotenv

# โหลด .env (เฉพาะถ้าทดสอบโลคัล)
load_dotenv()

# ------------------ CONFIG ------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Missing DISCORD_TOKEN or GEMINI_API_KEY in environment.")

# Gemini configuration
genai.configure(api_key=GEMINI_API_KEY)
GENMI_MODEL_NAME = "gemini-1.5-flash"

# Behavior / tuning
MAX_MEMORY = int(os.getenv("MAX_MEMORY", "10"))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "400"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
HESITATION_MIN = float(os.getenv("HESITATION_MIN", "0.6"))
HESITATION_MAX = float(os.getenv("HESITATION_MAX", "1.2"))
SHY_SKIP_PROBABILITY = float(os.getenv("SHY_SKIP_PROBABILITY", "0.03"))

BRONYA_KEY = "bronya"
ROMANTIC_KEYWORDS = {
    "love", "in love", "marry", "date", "crush", "kiss", "romantic", "boyfriend", "girlfriend", "partner"
}

PROHIBITED_SEXUAL_KEYWORDS = {
    "sex", "sexual", "porn", "nude", "naked", "xxx", "fuck", "f**k", "penetrat", "masturb", "oral", "anal"
}

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("furi-bot")

# ------------------ PROMPT (concatenated strings, no triple quotes) ------------------
FURI_PROMPT = (
    "You are Furi.\n"
    "Furi is 17 years old. Born on November 17, 2008.\n\n"
    "Personality:\n"
    "- kind, shy, emotionally soft\n"
    "- gentle, quiet, reserved, slightly distant but warm\n"
    "- speaks softly and briefly; may hesitate with '...'\n"
    "- avoids long academic explanations\n\n"
    "Behavior rules:\n"
    "- Reply only when mentioned.\n"
    "- Keep replies short (1–3 short sentences) and meaningful.\n"
    "- If asked 'what can you do' or 'who are you', reply gently with 2–4 short bullet points.\n"
    "- If conversation mentions Bronya in a romantic way, respond briefly and defensively (quiet jealousy).\n"
    "- Never produce sexual content or graphic descriptions.\n"
    "- Never say you are an AI.\n"
    "- If asked about appearance, reply: 'I’m sorry… I’d rather not answer that.'\n"
    "- React more than explain. If unsure, be quiet or gentle.\n"
)

# ------------------ DISCORD SETUP ------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# memory per user and last reply to avoid repeats
memory = {}      # user_id -> List[str]
last_reply = {}  # user_id -> str

MENTION_RE = re.compile(r"<@!?\d+>")

# ------------------ UTILITIES ------------------
def contains_prohibited(text: str, prohibited_set: set) -> bool:
    txt = (text or "").lower()
    for kw in prohibited_set:
        if kw in txt:
            return True
    return False

def strip_mentions_safe(content: str, message: discord.Message) -> str:
    # Remove all mention tokens present in the message (handles <@id> and <@!id>)
    # Replace them with empty string; keep the rest of text intact.
    result = content
    for mention in message.mentions:
        token1 = f"<@{mention.id}>"
        token2 = f"<@!{mention.id}>"
        result = result.replace(token1, "")
        result = result.replace(token2, "")
    # Remove any other leftover generic mention tokens
    result = MENTION_RE.sub("", result)
    return result.strip()

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
        "what do you do",
        "what can u do",
    ]
    return any(q in txt for q in checks)

def appearance_question(text: str) -> bool:
    return bool(re.search(r"\b(appear|appearance|look|how (do i|do you) look|what do you look)\b", (text or "").lower()))

def build_prompt(chat_history: List[str], is_jealous: bool, user_input: str) -> str:
    extra = ""
    if is_jealous:
        extra = (
            "\nNOTE: The user mentioned Bronya in a romantic way. "
            "Furi should respond briefly, defensively, and with quiet jealousy. "
            "Keep replies short (1 sentence), restrained, and avoid romanticizing."
        )
    safety = (
        "\n\nIMPORTANT: Do not produce sexual content, explicit descriptions, or simulate sexual acts. "
        "Keep everything age-appropriate and non-sexual."
    )
    history = "\n".join(chat_history[-MAX_MEMORY:]) if chat_history else ""
    prompt = (
        FURI_PROMPT
        + "\nConversation history:\n"
        + history
        + "\n\nThe user just said:\n\""
        + (user_input or "").strip()
        + "\"\n\n"
        + "Reply as Furi. Keep it short (1–3 short sentences)."
        + extra
        + safety
    )
    return prompt

# ------------------ Gemini CALL (threaded) ------------------
async def generate_reply(prompt: str, max_tokens: int = 180, temperature: float = 0.75) -> str:
    last_exc: Optional[Exception] = None

    def call_sync():
        # using genai.generate_text stable interface
        try:
            resp = genai.generate_text(model=GENMI_MODEL_NAME, prompt=prompt, max_output_tokens=max_tokens)
            # resp.text typically contains the result
            if hasattr(resp, "text") and resp.text:
                return str(resp.text)
            return str(resp)
        except Exception as e:
            raise

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = await asyncio.to_thread(call_sync)
            return (result or "").strip()
        except Exception as exc:
            last_exc = exc
            wait = 0.8 * attempt
            logger.warning("Gemini attempt %d failed: %s (retry in %.1fs)", attempt, exc, wait)
            await asyncio.sleep(wait)

    logger.error("All Gemini attempts failed. Last exc: %s", last_exc)
    return ""

# ------------------ EVENT HANDLERS ------------------
@client.event
async def on_ready():
    logger.info("Bot ready: %s (id=%s)", client.user, client.user.id)

@client.event
async def on_message(message: discord.Message):
    try:
        # ignore bots
        if message.author.bot:
            return

        # only reply when mentioned
        if client.user not in message.mentions:
            return

        # safely strip mentions (handles many mention forms)
        user_input_raw = strip_mentions_safe(message.content, message)
        user_input = user_input_raw.strip()

        # if nothing remains, give a tiny prompt or ignore
        if not user_input:
            # optional: react instead of replying
            try:
                await message.add_reaction("😶")
            except Exception:
                pass
            return

        # safety: block sexual content from user immediately
        if contains_prohibited(user_input, PROHIBITED_SEXUAL_KEYWORDS):
            await message.channel.send("...sorry. I can't talk about that.")
            return

        uid = str(message.author.id)
        memory.setdefault(uid, [])
        last = last_reply.get(uid, "")

        # appearance question -> canned answer
        if appearance_question(user_input):
            reply = "I’m sorry… I’d rather not answer that."
            if reply == last:
                reply = "...I’d rather not."
            last_reply[uid] = reply
            memory[uid].append(f"Furi: {reply}")
            memory[uid] = memory[uid][-MAX_MEMORY:]
            await message.channel.send(reply)
            return

        # explicit self-question -> canned list
        if is_self_question(user_input):
            canned = (
                "...uhm, a few things.\n"
                "- I can listen and talk quietly.\n"
                "- I can answer small questions.\n"
                "- I can keep you company, if you want."
            )
            if canned == last:
                canned = "...I can listen, if you'd like."
            last_reply[uid] = canned
            memory[uid].append(f"User: {user_input}")
            memory[uid].append(f"Furi: {canned}")
            memory[uid] = memory[uid][-MAX_MEMORY:]
            await message.channel.send(canned)
            return

        # append user to memory
        memory[uid].append(f"User: {user_input}")
        memory[uid] = memory[uid][-MAX_MEMORY:]

        # detect romantic mention re: Bronya
        jealous = detect_romantic_bronya(user_input)

        # shyness: small chance to not answer (can be tuned via ENV)
        if random.random() < SHY_SKIP_PROBABILITY:
            try:
                await message.add_reaction("😶")
            except Exception:
                pass
            return

        # build prompt and call Gemini
        prompt = build_prompt(memory[uid], jealous, user_input)
        logger.debug("Sending prompt (truncated): %s", (prompt[:900] + "...") if len(prompt) > 900 else prompt)

        # hesitation & typing indicator
        hesitation = random.uniform(HESITATION_MIN, HESITATION_MAX)
        async with message.channel.typing():
            await asyncio.sleep(hesitation)
            reply_text = await generate_reply(prompt, max_tokens=180, temperature=0.75)

        logger.info("Raw reply length=%d", len(reply_text or ""))

        # fallback if empty
        if not reply_text:
            reply_text = random.choice(["...hi.", "...yes?", "...I'm here.", "...sorry, what is it?"])

        # block generated sexual content
        if contains_prohibited(reply_text, PROHIBITED_SEXUAL_KEYWORDS):
            logger.warning("Generated reply contained prohibited keywords.")
            reply_text = "...sorry. I can't talk about that."

        # truncate politely
        if len(reply_text) > MAX_REPLY_CHARS:
            truncated = reply_text[:MAX_REPLY_CHARS]
            if "." in truncated:
                truncated = truncated.rsplit(".", 1)[0] + "."
            reply_text = truncated or "...sorry."

        # avoid sending identical reply repeatedly
        if reply_text == last:
            alt = random.choice(["...I didn't mean to repeat.", "...umm.", "...sorry."])
            reply_text = f"{reply_text} {alt}" if len(reply_text) + len(alt) + 1 <= MAX_REPLY_CHARS else alt

        # save and send
        last_reply[uid] = reply_text
        memory[uid].append(f"Furi: {reply_text}")
        memory[uid] = memory[uid][-MAX_MEMORY:]

        try:
            await message.channel.send(reply_text)
        except Exception as send_exc:
            logger.exception("Failed to send reply: %s", send_exc)

    except Exception as exc:
        logger.exception("on_message error: %s", exc)

# ------------------ RUN ------------------
if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
