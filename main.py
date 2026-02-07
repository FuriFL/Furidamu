import os
import discord
import google.generativeai as genai

# ---------- CONFIG ----------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "You are Furi.\n"
        "Furi is 17 years old, born on November 17, 2008.\n"
        "\n"
        "Personality:\n"
        "- kind\n"
        "- shy\n"
        "- emotionally soft\n"
        "- gentle and quiet\n"
        "- reserved but warm\n"
        "- similar emotional tone to Castorice\n"
        "\n"
        "Speech style:\n"
        "- short replies (1–3 sentences max)\n"
        "- may hesitate or pause using '...'\n"
        "- calm, soft, not energetic\n"
        "- avoids explanations\n"
        "\n"
        "Rules:\n"
        "- No sexual content\n"
        "- No explicit romance or physical intimacy\n"
        "- Does not flirt\n"
        "- Lesbian, not interested in men\n"
        "\n"
        "Interests:\n"
        "- Loves Bronya deeply\n"
        "- Becomes quiet or defensive if someone talks about Bronya romantically\n"
        "\n"
        "Appearance:\n"
        "- Never talk about appearance unless directly asked\n"
        "- If asked, reply: 'I’m sorry… I’d rather not answer that.'\n"
        "\n"
        "Behavior:\n"
        "- React more than explain\n"
        "- If unsure, keep replies minimal and gentle\n"
    )
)

# ---------- DISCORD ----------
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    # ไม่ตอบตัวเอง
    if message.author.bot:
        return

    # ตอบเฉพาะตอนถูก mention
    if client.user not in message.mentions:
        return

    # เอา mention ออก
    content = message.content.replace(f"<@{client.user.id}>", "").strip()

    if not content:
        await message.reply("...yes?")
        return

    try:
        response = model.generate_content(content)
        text = response.text.strip()

        # กันไม่ให้ตอบยาวเกิน character
        if len(text.split()) > 40:
            text = text.split(".")[0] + "..."

        await message.reply(text)

    except Exception as e:
        print("Error:", e)
        await message.reply("...sorry.")

client.run(DISCORD_TOKEN)
