from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Makkah Ultra-Loop 5000 - World Series")

@app.get("/", response_class=HTMLResponse)
async def serve_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
