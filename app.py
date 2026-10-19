import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Crochet Studio Engine")

SYMBOLS = ["■", "●", "▲", "◆", "★", "✚", "✖", "▼", "◀", "▶", "♥", "♠", "♣", "◉", "▣", "✦", "✧", "✪", "✫", "✬"]


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2])).upper()


def clean_orphan_stitches(grid: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Supprime les mailles isolées ('orphelines') pénibles à crocheter
    en remplaçant tout pixel qui a 0 ou 1 voisin identique par la couleur majoritaire autour de lui.
    """
    if iterations <= 0:
        return grid
    h, w = grid.shape
    cleaned = grid.copy()
    for _ in range(iterations):
        padded = np.pad(cleaned, 1, mode="edge")
        neighbors = np.stack([
            padded[0:h, 0:w],     padded[0:h, 1:w+1],     padded[0:h, 2:w+2],
            padded[1:h+1, 0:w],                           padded[1:h+1, 2:w+2],
            padded[2:h+2, 0:w],   padded[2:h+2, 1:w+1],   padded[2:h+2, 2:w+2]
        ], axis=-1)

        new_grid = cleaned.copy()
        for r in range(h):
            for c in range(w):
                val = cleaned[r, c]
                nb = neighbors[r, c]
                if np.sum(nb == val) <= 1:
                    vals, counts = np.unique(nb, return_counts=True)
                    new_grid[r, c] = vals[np.argmax(counts)]
        cleaned = new_grid
    return cleaned


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/generate")
async def generate_pattern(
    file: UploadFile = File(...),
    width: int = Form(60),
    n_colors: int = Form(6),
    smooth_level: int = Form(1),
    stitch_ratio: float = Form(1.0)
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Gestion de la transparence (fond blanc par défaut pour les PNG détourés)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            alpha = img.convert("RGBA")
            bg = Image.new("RGBA", alpha.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, alpha).convert("RGB")
        else:
            img = img.convert("RGB")

        width = max(10, min(200, width))
        n_colors = max(2, min(20, n_colors))
        smooth_level = max(0, min(3, smooth_level))

        # Calcul de la hauteur en tenant compte du ratio de la maille
        aspect = img.height / img.width
        height = max(10, int(round(width * aspect * stitch_ratio)))

        # Redimensionnement haute qualité
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
        data = np.array(img_resized, dtype=np.float32)
        pixels = data.reshape(-1, 3)

        # Quantification des couleurs par K-Means
        kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        # Tri des couleurs de la plus sombre à la plus claire (luminance perçue)
        luminance = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
        sorted_order = np.argsort(luminance)
        remap = np.zeros_like(sorted_order)
        remap[sorted_order] = np.arange(n_colors)

        sorted_centers = centers[sorted_order]
        grid = remap[labels].reshape(height, width)

        # Nettoyage des mailles isolées
        if smooth_level > 0:
            grid = clean_orphan_stitches(grid, iterations=smooth_level)

        palette = []
        for idx, rgb in enumerate(sorted_centers):
            code = chr(65 + idx) if idx < 26 else f"C{idx+1}"
            palette.append({
                "id": idx,
                "code": code,
                "name": f"Pelote {code}",
                "hex": rgb_to_hex(rgb),
                "symbol": SYMBOLS[idx % len(SYMBOLS)]
            })

        return JSONResponse({
            "width": width,
            "height": height,
            "grid": grid.tolist(),
            "palette": palette
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du traitement de l'image : {str(e)}")
