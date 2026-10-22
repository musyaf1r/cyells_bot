from groq import Groq

# Ganti dengan API key kamu
GROQ_API_KEY = "GROQ_API_KEY"
client = Groq(api_key=GROQ_API_KEY)

models = client.models.list()
print("=== Semua model yang tersedia di akun kamu ===")
for m in sorted(models.data, key=lambda x: x.id):
    print(" -", m.id)

