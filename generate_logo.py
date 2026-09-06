import math
from PIL import Image, ImageDraw, ImageFont

def create_unique_smart_kishan_logo(output_filename="smart_kishan_logo.png"):
    # 1000x1000 High-Res Canvas with Transparent Background
    size = (1000, 1000)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    center_x, center_y = 500, 360

    # 1. Subtle Radial Glow Background
    for r in range(320, 180, -8):
        alpha = int(22 * (1 - (r - 180) / 140))
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], 
                     fill=(232, 245, 233, alpha))

    # 2. Modern Open Agro-Shield Arc
    # Gradient Blue-Green Protective Outer Ring
    for w in range(16):
        bbox = [center_x - 240 + w, center_y - 240 + w, center_x + 240 - w, center_y + 240 - w]
        draw.arc(bbox, start=35, end=290, fill=(46, 125, 50, 240), width=2)
        draw.arc(bbox, start=120, end=350, fill=(21, 101, 192, 220), width=2)

    # 3. Precision Sensor / Wireless Telemetry Waves (Top Center)
    wave_center_y = center_y - 170
    draw.arc([center_x - 55, wave_center_y - 30, center_x + 55, wave_center_y + 40], 
             start=200, end=340, fill=(2, 136, 209, 255), width=7)
    draw.arc([center_x - 35, wave_center_y - 12, center_x + 35, wave_center_y + 35], 
             start=200, end=340, fill=(2, 136, 209, 255), width=6)
    draw.ellipse([center_x - 8, wave_center_y + 12, center_x + 8, wave_center_y + 28], 
                 fill=(255, 179, 0, 255)) # Gold sensor core

    # 4. Leaf 1: Organic Natural Leaf (Left side)
    organic_leaf_points = [
        (center_x - 10, center_y + 130),
        (center_x - 80, center_y + 100),
        (center_x - 160, center_y + 10),
        (center_x - 160, center_y - 80),
        (center_x - 110, center_y - 140),
        (center_x - 40, center_y - 90),
        (center_x - 15, center_y - 10),
        (center_x - 5, center_y + 80),
    ]
    draw.polygon(organic_leaf_points, fill=(67, 160, 71, 245), outline=(27, 94, 32, 255))
    
    # Internal Veins
    draw.line([(center_x - 110, center_y - 140), (center_x - 10, center_y + 130)], fill=(165, 214, 167, 230), width=5)
    draw.line([(center_x - 70, center_y - 30), (center_x - 130, center_y - 40)], fill=(165, 214, 167, 200), width=3)
    draw.line([(center_x - 50, center_y + 30), (center_x - 110, center_y + 40)], fill=(165, 214, 167, 200), width=3)

    # 5. Leaf 2: Digital IoT / Neural Circuit Leaf (Right side)
    digital_leaf_points = [
        (center_x - 5, center_y + 140),
        (center_x + 30, center_y + 90),
        (center_x + 120, center_y + 30),
        (center_x + 150, center_y - 60),
        (center_x + 100, center_y - 160),
        (center_x + 40, center_y - 100),
        (center_x + 5, center_y - 10),
        (center_x - 5, center_y + 70),
    ]
    draw.polygon(digital_leaf_points, fill=(2, 119, 189, 245), outline=(1, 87, 155, 255))

    # Circuit Board Nodes & Bus Lines
    circuit_nodes = [
        ((center_x + 45, center_y - 50), (center_x + 85, center_y - 80)),
        ((center_x + 45, center_y - 50), (center_x + 55, center_y + 10)),
        ((center_x + 55, center_y + 10), (center_x + 105, center_y + 5)),
        ((center_x + 55, center_y + 10), (center_x + 35, center_y + 75)),
        ((center_x + 35, center_y + 75), (center_x + 85, center_y + 65)),
    ]
    for start, end in circuit_nodes:
        draw.line([start, end], fill=(179, 229, 252, 230), width=4)
        draw.ellipse([start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6], fill=(255, 255, 255, 255))
        draw.ellipse([end[0] - 6, end[1] - 6, end[0] + 6, end[1] + 6], fill=(255, 215, 0, 255))

    # 6. Professional Typography Rendering
    try:
        font_main = ImageFont.truetype("arialbd.ttf", 82)
        font_sub = ImageFont.truetype("arialbd.ttf", 29)
    except Exception:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # "SMART" in Deep Royal Navy Blue (#1565C0)
    # "KISHAN" in Agricultural Forest Green (#2E7D32)
    text_y = 690
    draw.text((220, text_y), "SMART", font=font_main, fill=(21, 101, 192, 255))
    draw.text((535, text_y), "KISHAN", font=font_main, fill=(46, 125, 50, 255))

    # Sub-heading banner
    draw.text((250, 795), "DIGITAL  FARMING  SOLUTIONS", font=font_sub, fill=(71, 85, 105, 255))

    # Save output file
    img.save(output_filename, format="PNG")
    print(f"Generated clean logo: {output_filename}")

if __name__ == "__main__":
    create_unique_smart_kishan_logo("smart_kishan_logo.png")
