import math
from PIL import Image, ImageDraw, ImageFont

def create_agricultural_field_stamp(output_filename="smart_kishan_stamp.png"):
    # 500x500 High-Resolution Circular Agricultural Seal
    size = (500, 500)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    cx, cy, radius = 250, 250, 225

    # 1. Rich Forest & Light Green Agricultural Background Circle
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(27, 94, 32, 255))
    
    # Inner light green meadow gradient ring
    draw.ellipse([cx - radius + 12, cy - radius + 12, cx + radius - 12, cy + radius - 12], fill=(46, 125, 50, 255))

    # 2. Double Crisp White Border Rings
    draw.ellipse([cx - radius + 20, cy - radius + 20, cx + radius - 20, cy + radius - 20], outline=(255, 255, 255, 255), width=3)
    draw.ellipse([cx - radius + 27, cy - radius + 27, cx + radius - 27, cy + radius - 27], outline=(255, 215, 0, 255), width=2)

    # 3. Agricultural Field Graphics (Sun, Hills, Crops inside stamp center)
    # Golden Sun
    draw.ellipse([cx - 35, cy - 85, cx + 35, cy - 15], fill=(255, 215, 0, 255))
    
    # Green Rolling Farm Hills
    draw.arc([cx - 120, cy - 40, cx + 120, cy + 80], start=180, end=360, fill=(129, 199, 132, 255), width=18)
    draw.arc([cx - 90, cy - 10, cx + 90, cy + 100], start=180, end=360, fill=(102, 187, 106, 255), width=14)

    # Growing Crop Shoots
    for dx in [-40, -20, 0, 20, 40]:
        bx = cx + dx
        by = cy + 25
        draw.polygon([(bx, by), (bx - 5, by - 22), (bx + 5, by - 22)], fill=(200, 230, 201, 255))

    try:
        f_top = ImageFont.truetype("arialbd.ttf", 26)
        f_mid = ImageFont.truetype("arialbd.ttf", 36)
        f_bot = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        f_top = f_mid = f_bot = ImageFont.load_default()

    # 4. Text Layout
    draw.text((cx - 118, cy - 125), "GOVT COMPLIANT", font=f_top, fill=(255, 255, 255, 255))
    
    # Golden SMART KISHAN Center Banner
    draw.text((cx - 132, cy + 48), "SMART KISHAN", font=f_mid, fill=(255, 215, 0, 255))
    
    # Bottom 4R Certified text
    draw.text((cx - 105, cy + 95), "★ 4R CERTIFIED ★", font=f_bot, fill=(255, 255, 255, 255))

    img.save(output_filename, format="PNG")
    print(f"Generated agricultural stamp: {output_filename}")


if __name__ == "__main__":
    create_agricultural_field_stamp("smart_kishan_stamp.png")
