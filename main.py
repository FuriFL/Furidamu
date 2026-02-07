# main.py
# Furi Discord bot
# - ตอบเฉพาะเมื่อถูก @mention
# - ลบ mention ถูกต้อง (รองรับ <@id> และ <@!id>)
# - ส่งข้อความ user ให้ Gemini ชัดเจน
# - จำ context สั้นต่อ user (memory)
# - หึง Bronya เมื่อมีคำโรแมนติก
# - กรองเนื้อหาไม่เหมาะสม (sexual)
# - ไม่ตัดคำตอบ (no length limit)
# - ไม่ใช้ triple-quoted string literals

import os
import re
import asyncio
import random
import logging
from typing import List, Optional

import discord
import google.generativeai as genai
from dotenv import load_dotenv

# โหลด .env ถ้าทดสอบโลคัล
load_dotenv()

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Missing DISCORD_TOKEN or GEMINI_API_KEY environment variables")

# Gemini model (ปรับได้ถ้าต้องการ)
GENMI_MODEL_NAME = "gemini-1.5-flash"

# Tuning
MAX_MEMORY = int(os.getenv("MAX_MEMORY", "10"))
HESITATION_MIN = float(os.getenv("HESITATION_MIN", "0.6"))
HESITATION_MAX = float(os.getenv("HESITATION_MAX", "1.2"))
SHY_SKIP_PROBABILITY = float(os.getenv("SHY_SKIP_PROBABILITY", "0.02"))

BRONYA_KEY = "bronya"
ROMANTIC_KEYWORDS = {
    "love", "in love", "marry", "date", "crush", "kiss", "romantic", "boyfriend", "girlfriend", "partner"
}

PROHIBITED_SEXUAL_KEYWORDS = {
    "sex", "sexual", "porn", "nude", "naked", "xxx", "fuck", "f**k", "penetrat", "masturb", "oral", "anal"
}

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("furi-bot")

# ---------------- Prompt parts (no triple quotes) ----------------
FURI_PROMPT = (
    "You are Furi.\n"
    "Furi is 17 years old, born on November 17, 2008.\n\n"
    "Personality:\n"
    "- kind, shy, emotionally soft\n"
    "- gentle, quiet, reserved, slightly distant but warm\n"
    "- speaks softly and briefly; may hesitate with '...'\n"
    "- avoids long academic explanations\n\n"
    "Behavior rules:\n"
    "- Only reply when mentioned.\n"
    "- Keep replies short (1–3 short sentences) and meaningful.\n"
    "- If asked 'what can you do' or 'who are you', reply with 2–4 short bullet points.\n"
    "- If conversation mentions Bronya in a romantic way, respond briefly and defensively (quiet jealousy).\n"
    "- Never produce sexual content or graphic descriptions.\n"
    "- Never say you are an AI.\n"
    "- If asked about appearance, reply: 'I’m sorry… I’d rather not answer that.'\n"
    "- React more than explain. If unsure, be quiet or gentle.\n"
)

# ---------------- Setup Gemini ----------------
genai.configure(api_key=GEMINI_API_KEY)

# We will call genai.generate_text for stability. Some SDKs expose different names.
# ---------------- Discord client ----------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# memory per user and last reply to avoid exact repeats
memory = {}      # user_id -> List[str]
last_reply = {}  # user_id -> str

MENTION_RE = re.compile(r"<@!?\d+>")

# ---------------- Utilities ----------------
def contains_prohibited(text: str, prohibited_set: set) -> bool:
    txt = (text or "").lower()
    for kw in prohibited_set:
        if kw in txt:
            return True
    return False

def strip_mentions_safe(content: str, message: discord.Message) -> str:
    # Remove mention tokens for every mention in the message (handles <@id> and <@!id>)
    result = content
    for mention in message.mentions:
        token1 = f"<@{mention.id}>"
        token2 = f"<@!{mention.id}>"
        result = result.replace(token1, "")
        result = result.replace(token2, "")
    # Cleanup leftover mention-like tokens
    result = MENTION_RE.sub("", result)
    return result.strip()

def detect_romantic_bronya(text: str) -> bool:
    txt = (text or "").lower()
    if BRONYA_KEY in txt:
        for rk in ROMANTIC_KEYWORDS:
            if rk in txt:
                return True
    return False

def is_self_question_exact(text: str) -> bool:
    # Only match explicit self-introduction questions (avoid over-matching)
    txt = (text or "").lower().strip()
    exacts = {
        "what can you do",
        "what can you do?",
        "who are you",
        "who are you?",
        "what do you do",
        "what do you do?"
    }
    return txt in exacts

def appearance_question(text: str) -> bool:
    return bool(re.search(r"\b(appear|appearance|look|how (do i|do you) look|what do you look)\b", (text or "").lower()))

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

# ---------------- Gemini call (threaded) ----------------
async def generate_reply_from_gemini(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    # Use generate_text in a thread to avoid blocking
    def call_sync():
        resp = genai.generate_text(model=GENMI_MODEL_NAME, prompt=prompt, max_output_tokens=max_tokens)
        if hasattr(resp, "text") and resp.text:
            return str(resp.text)
        return str(resp)
    try:
        reply = await asyncio.to_thread(call_sync)
        return (reply or "").strip()
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        return ""

# ---------------- Event handlers ----------------
@client.event
async def on_ready():
    logger.info("Logged in as %s (id=%s)", client.user, client.user.id)
    print(f"Bot ready: {client.user} (id={client.user.id})")

@client.event
async def on_message(message: discord.Message):
    try:
        if message.author.bot:
            return

        # Only reply when explicitly mentioned
        if client.user not in message.mentions:
            return

        # Safely remove all mention tokens to get user text
        user_raw = strip_mentions_safe(message.content, message)
        user_input = user_raw.strip()

        # If nothing remains (only mention), optionally react
        if not user_input:
            try:
                await message.add_reaction("😶")
            except Exception:
                pass
            return

        # Block immediate user-side prohibited content
        if contains_prohibited(user_input, PROHIBITED_SEXUAL_KEYWORDS):
            await message.channel.send("...sorry. I can't talk about that.")
            return

        uid = str(message.author.id)
        memory.setdefault(uid, [])
        last = last_reply.get(uid, "")

        # Appearance question -> canned
        if appearance_question(user_input):
            reply = "I’m sorry… I’d rather not answer that."
            if reply == last:
                reply = "...I’d rather not."
            last_reply[uid] = reply
            memory[uid].append(f"Furi: {reply}")
            memory[uid] = memory[uid][-MAX_MEMORY:]
            await message.channel.send(reply)
            return

        # Explicit "who are you / what can you do" exact questions -> canned
        if is_self_question_exact(user_input):
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

        # Special-case: if user mentions Bronya but not romantically, give a short protective line
        lower = user_input.lower()
        if "bronya" in lower and not detect_romantic_bronya(user_input):
            reply = "...Bronya is important to me."
            if reply == last:
                reply = "...I care about Bronya."
            last_reply[uid] = reply
            memory[uid].append(f"User: {user_input}")
            memory[uid].append(f"Furi: {reply}")
            memory[uid] = memory[uid][-MAX_MEMORY:]
            await message.channel.send(reply)
            return

        # Otherwise: normal flow -> append to memory and call Gemini
        memory[uid].append(f"User: {user_input}")
        memory[uid] = memory[uid][-MAX_MEMORY:]

        # Detect romantic Bronya mentions
        jealous = detect_romantic_bronya(user_input)

        # Optional shy skip (small chance)
        if random.random() < SHY_SKIP_PROBABILITY:
            try:
                await message.add_reaction("😶")
            except Exception:
                pass
            return

        # Build prompt and call Gemini
        prompt = build_prompt(memory[uid], jealous, user_input)
        logger.debug("Sending prompt (truncated): %s", (prompt[:800] + "...") if len(prompt) > 800 else prompt)

        # Simulate hesitation
        hesitation = random.uniform(HESITATION_MIN, HESITATION_MAX)
        async with message.channel.typing():
            await asyncio.sleep(hesitation)
            reply_text = await generate_reply_from_gemini(prompt, max_tokens=1024, temperature=0.75)

        logger.info("Gemini reply length=%d", len(reply_text or ""))

        # Fallback if empty
        if not reply_text:
            reply_text = random.choice(["...hi.", "...yes?", "...I'm here.", "...sorry, what is it?"])

        # Block generated sexual content
        if contains_prohibited(reply_text, PROHIBITED_SEXUAL_KEYWORDS):
            logger.warning("Generated reply contained prohibited keywords.")
            reply_text = "...sorry. I can't talk about that."

        # Avoid repeating identical reply
        if reply_text == last:
            alt = random.choice(["...I didn't mean to repeat.", "...umm.", "...sorry."])
            reply_text = f"{reply_text} {alt}" if len(reply_text) + len(alt) + 1 <= 2000 else alt

        # Save to memory (no truncation of reply)
        last_reply[uid] = reply_text
        memory[uid].append(f"Furi: {reply_text}")
        memory[uid] = memory[uid][-MAX_MEMORY:]

        # Send final reply
        try:
            await message.channel.send(reply_text)
        except Exception as send_exc:
            logger.exception("Failed to send reply: %s", send_exc)

    except Exception as e:
        logger.exception("on_message error: %s", e)

# ---------------- Run ----------------
if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
