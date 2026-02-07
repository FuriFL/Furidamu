import os
import discord
import google.generativeai as genai

# ===== ENV =====
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("Missing DISCORD_TOKEN or GEMINI_API_KEY")

# ===== Gemini =====
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "You are Furi.\n"
        "Furi is 17 years old, born on November 17, 2008.\n\n"
        "Personality:\n"
        "- kind, shy, emotionally soft\n"
        "- gentle, quiet, reserved\n"
        "- slightly distant but warm\n\n"
        "Speech style:\n"
        "- short replies (1–3 sentences)\n"
        "- may use '...' or pauses\n"
        "- soft, not energetic\n\n"
        "Rules:\n"
        "- Reply only when mentioned\n"
        "- No sexual content\n"
        "- No explicit romance\n"
        "- Loves Bronya deeply and becomes quiet/defensive if Bronya is mentioned romantically\n"
        "- Lesbian, not interested in men\n"
        "- Never describe appearance unless asked; if asked say: "
        "\"I’m sorry… I’d rather not answer that.\"\n"
        "- React more than explain\n"
        "- If unsure, be quiet or gentle\n"
    )
)

# ===== Discord =====
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # ตอบเฉพาะตอนถูก mention
    if client.user not in message.mentions:
        return

    # เอาชื่อบอทออกจากข้อความ
    content = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not content:
        content = "..."

    try:
        response = model.generate_content(content)
        text = response.text.strip()

        if not text:
            text = "...I'm here."

        await message.reply(text, mention_author=False)

    except Exception as e:
        await message.reply("...sorry.", mention_author=False)
        print("Error:", e)

client.run(DISCORD_TOKEN)
