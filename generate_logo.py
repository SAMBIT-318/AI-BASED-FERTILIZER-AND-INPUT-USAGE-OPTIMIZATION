import math
from PIL import Image, ImageDraw


def create_agriculture_logo(output_filename="smart_kishan_agriculture_logo.png"):

    # ============================================================
    # 1. CANVAS
    # ============================================================
    size = 1000
    img = Image.new("RGBA", (size, size), (250, 249, 240, 255))
    draw = ImageDraw.Draw(img)

    cx, cy = 500, 500
    outer_r = 390

    # ============================================================
    # 2. OUTER GREEN CIRCULAR LOGO
    # ============================================================

    # Main circular border
    draw.ellipse(
        [
            cx - outer_r,
            cy - outer_r,
            cx + outer_r,
            cy + outer_r
        ],
        fill=(20, 105, 48, 255)
    )

    # Inner white circular border
    inner_r = 365

    draw.ellipse(
        [
            cx - inner_r,
            cy - inner_r,
            cx + inner_r,
            cy + inner_r
        ],
        fill=(255, 255, 255, 255)
    )

    # Inner landscape circle
    landscape_r = 345

    draw.ellipse(
        [
            cx - landscape_r,
            cy - landscape_r,
            cx + landscape_r,
            cy + landscape_r
        ],
        fill=(224, 242, 230, 255)
    )

    # ============================================================
    # 3. SKY
    # ============================================================

    # Soft blue sky gradient
    top_y = cy - landscape_r
    bottom_y = 560

    for y in range(top_y, bottom_y):

        ratio = (y - top_y) / (bottom_y - top_y)

        r = int(190 + (240 - 190) * ratio)
        g = int(225 + (248 - 225) * ratio)
        b = int(242 + (220 - 242) * ratio)

        draw.line(
            [(cx - landscape_r, y),
             (cx + landscape_r, y)],
            fill=(r, g, b, 255)
        )

    # ============================================================
    # 4. CLOUDS
    # ============================================================

    cloud_color = (245, 249, 242, 220)

    draw.ellipse([190, 365, 290, 425], fill=cloud_color)
    draw.ellipse([235, 340, 340, 425], fill=cloud_color)
    draw.ellipse([300, 370, 390, 425], fill=cloud_color)

    draw.ellipse([700, 370, 790, 425], fill=cloud_color)
    draw.ellipse([750, 345, 850, 425], fill=cloud_color)
    draw.ellipse([810, 370, 890, 425], fill=cloud_color)

    # ============================================================
    # 5. SUN
    # ============================================================

    sun_x = 500
    sun_y = 475
    sun_r = 82

    # Sun rays
    for angle in range(0, 360, 30):

        rad = math.radians(angle)

        x1 = sun_x + math.cos(rad) * 105
        y1 = sun_y + math.sin(rad) * 105

        x2 = sun_x + math.cos(rad) * 145
        y2 = sun_y + math.sin(rad) * 145

        draw.line(
            [(x1, y1), (x2, y2)],
            fill=(245, 173, 20, 255),
            width=12
        )

    # Sun
    draw.ellipse(
        [
            sun_x - sun_r,
            sun_y - sun_r,
            sun_x + sun_r,
            sun_y + sun_r
        ],
        fill=(255, 190, 25, 255)
    )

    # ============================================================
    # 6. DISTANT GREEN HILLS
    # ============================================================

    # Back hill
    draw.polygon(
        [
            (155, 560),
            (270, 510),
            (375, 540),
            (500, 485),
            (625, 535),
            (750, 495),
            (860, 550),
            (860, 650),
            (155, 650)
        ],
        fill=(122, 190, 105, 255)
    )

    # Front hill
    draw.polygon(
        [
            (155, 585),
            (300, 545),
            (420, 580),
            (560, 525),
            (690, 565),
            (860, 535),
            (860, 680),
            (155, 680)
        ],
        fill=(61, 145, 65, 255)
    )

    # ============================================================
    # 7. TREES
    # ============================================================

    def draw_tree(x, y, scale=1):

        trunk_w = int(14 * scale)
        trunk_h = int(55 * scale)

        draw.rectangle(
            [
                x - trunk_w // 2,
                y,
                x + trunk_w // 2,
                y + trunk_h
            ],
            fill=(74, 87, 45, 255)
        )

        # Tree crown
        draw.ellipse(
            [
                x - 35 * scale,
                y - 80 * scale,
                x + 35 * scale,
                y + 10 * scale
            ],
            fill=(38, 126, 55, 255)
        )

    draw_tree(710, 505, 1.0)
    draw_tree(770, 520, 1.15)
    draw_tree(830, 500, 0.9)

    # ============================================================
    # 8. AGRICULTURAL FIELD
    # ============================================================

    field_top = 610
    field_bottom = 850

    draw.polygon(
        [
            (155, field_top),
            (860, field_top),
            (930, field_bottom),
            (80, field_bottom)
        ],
        fill=(94, 173, 45, 255)
    )

    # Field rows
    row_color = (238, 249, 224, 255)

    rows = [
        [(230, 610), (430, 850)],
        [(330, 610), (490, 850)],
        [(430, 610), (550, 850)],
        [(530, 610), (620, 850)],
        [(630, 610), (690, 850)],
        [(730, 610), (760, 850)],
        [(820, 610), (830, 850)]
    ]

    for start, end in rows:

        draw.line(
            [start, end],
            fill=row_color,
            width=10
        )

    # ============================================================
    # 9. TRACTOR
    # ============================================================

    tractor_green = (18, 103, 53, 255)
    tractor_dark = (12, 74, 39, 255)
    tractor_white = (245, 245, 235, 255)

    # Tractor body
    draw.rounded_rectangle(
        [290, 545, 430, 625],
        radius=12,
        fill=tractor_green
    )

    # Tractor bonnet
    draw.rectangle(
        [395, 565, 470, 615],
        fill=tractor_green
    )

    # Tractor cabin frame
    draw.line(
        [(305, 550), (305, 490)],
        fill=tractor_dark,
        width=12
    )

    draw.line(
        [(305, 490), (380, 490)],
        fill=tractor_dark,
        width=12
    )

    draw.line(
        [(380, 490), (395, 555)],
        fill=tractor_dark,
        width=12
    )

    # Cabin roof
    draw.rectangle(
        [292, 480, 390, 500],
        fill=tractor_green
    )

    # Cabin glass
    draw.polygon(
        [
            (320, 505),
            (365, 505),
            (378, 550),
            (320, 550)
        ],
        fill=(210, 235, 220, 255)
    )

    # Rear wheel
    draw.ellipse(
        [270, 575, 345, 650],
        fill=tractor_dark
    )

    draw.ellipse(
        [288, 593, 327, 632],
        fill=tractor_white
    )

    draw.ellipse(
        [299, 604, 316, 621],
        fill=tractor_dark
    )

    # Front wheel
    draw.ellipse(
        [415, 585, 465, 635],
        fill=tractor_dark
    )

    draw.ellipse(
        [428, 598, 452, 622],
        fill=tractor_white
    )

    # Exhaust pipe
    draw.rectangle(
        [445, 515, 455, 565],
        fill=tractor_dark
    )

    draw.rectangle(
        [440, 510, 460, 520],
        fill=tractor_dark
    )

    # ============================================================
    # 10. FOREGROUND LEAVES
    # ============================================================

    leaf_dark = (12, 91, 43, 255)
    leaf_green = (65, 153, 38, 255)

    # Left large leaf
    draw.polygon(
        [
            (500, 920),
            (410, 820),
            (300, 780),
            (355, 860),
            (450, 915)
        ],
        fill=leaf_dark
    )

    # Left leaf vein
    draw.line(
        [(500, 920), (350, 810)],
        fill=(225, 242, 205, 255),
        width=8
    )

    # Right large leaf
    draw.polygon(
        [
            (500, 920),
            (590, 815),
            (700, 765),
            (650, 850),
            (555, 910)
        ],
        fill=leaf_green
    )

    # Right leaf vein
    draw.line(
        [(500, 920), (665, 800)],
        fill=(225, 242, 205, 255),
        width=8
    )

    # Central stem
    draw.line(
        [(500, 925), (500, 780)],
        fill=leaf_dark,
        width=12
    )

    # ============================================================
    # 11. TOP LEAF / CROWN
    # ============================================================

    top_leaf_dark = (16, 104, 48, 255)
    top_leaf_light = (71, 155, 35, 255)

    # Large top leaf
    draw.polygon(
        [
            (250, 300),
            (350, 185),
            (515, 135),
            (680, 160),
            (570, 235),
            (430, 290),
            (310, 310)
        ],
        fill=top_leaf_dark
    )

    # Leaf highlight
    draw.line(
        [(275, 285), (410, 220), (560, 175)],
        fill=(245, 250, 235, 255),
        width=12
    )

    # Right upper leaf
    draw.polygon(
        [
            (550, 240),
            (670, 190),
            (780, 245),
            (690, 300),
            (590, 310)
        ],
        fill=top_leaf_light
    )

    # ============================================================
    # 12. WHITE LOGO SEPARATION LINES
    # ============================================================

    # Curved-style separation between landscape and leaves
    draw.arc(
        [180, 470, 820, 800],
        start=180,
        end=360,
        fill=(255, 255, 255, 255),
        width=10
    )

    # ============================================================
    # 13. CLEAN OUTER RING
    # ============================================================

    draw.ellipse(
        [
            cx - outer_r,
            cy - outer_r,
            cx + outer_r,
            cy + outer_r
        ],
        outline=(10, 75, 38, 255),
        width=12
    )

    # ============================================================
    # 14. SAVE HIGH-QUALITY PNG
    # ============================================================

    img.save(
        output_filename,
        format="PNG",
        optimize=True
    )

    print(
        f"Agricultural logo successfully generated: "
        f"{output_filename}"
    )


# ================================================================
# RUN PROGRAM
# ================================================================

if __name__ == "__main__":
    create_agriculture_logo(
        "smart_kishan_agriculture_logo.png"
    )
