from typing import List
from PIL import Image, ImageDraw

from core.objects import extract_objects

ARC_PALETTE = {
    0: (0, 0, 0),         # black
    1: (0, 116, 217),     # blue
    2: (255, 65, 54),     # red
    3: (46, 204, 64),     # green
    4: (255, 220, 0),     # yellow
    5: (170, 170, 170),   # gray
    6: (240, 18, 190),    # magenta
    7: (255, 133, 27),    # orange
    8: (127, 219, 255),   # cyan
    9: (135, 12, 37),     # brown-ish
}

def render_grid(grid: List[List[int]], cell_size: int = 40, margin: int = 20) -> Image.Image:
    rows = len(grid)
    cols = len(grid[0])
    width = cols * cell_size + 2 * margin
    height = rows * cell_size + 2 * margin

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            color = ARC_PALETTE.get(grid[r][c], (100, 100, 100))
            x1 = margin + c * cell_size
            y1 = margin + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(80, 80, 80))

    return img

def overlay_objects(img: Image.Image, grid: List[List[int]], cell_size: int = 40, margin: int = 20) -> Image.Image:
    draw = ImageDraw.Draw(img)
    objects = extract_objects(grid)

    for obj in objects:
        min_r, min_c, max_r, max_c = obj.bbox
        x1 = margin + min_c * cell_size
        y1 = margin + min_r * cell_size
        x2 = margin + (max_c + 1) * cell_size
        y2 = margin + (max_r + 1) * cell_size

        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=3)

        cr, cc = obj.centroid
        cx = margin + int((cc + 0.5) * cell_size)
        cy = margin + int((cr + 0.5) * cell_size)
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(255, 255, 255))

        draw.text((x1 + 4, y1 + 4), f"id={obj.obj_id}", fill=(255, 255, 255))

    return img

def save_render(grid: List[List[int]], out_path: str, with_overlay: bool = True):
    img = render_grid(grid)
    if with_overlay:
        img = overlay_objects(img, grid)
    img.save(out_path)
