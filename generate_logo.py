import math
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# SMART KISHAN - PROFESSIONAL AGRICULTURE CERTIFICATION BADGE
# ============================================================

def get_font(size, bold=True):
    """
    Try several common fonts.
    """
    fonts = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    ]

    for font in fonts:
        try:
            return ImageFont.truetype(font, size)
        except:
            pass

    return ImageFont.load_default()


def center_text(draw, text, y, font, fill):
    """
    Draw horizontally centered text.
    """
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]

    draw.text(
        ((1000 - width) / 2, y),
        text,
        font=font,
        fill=fill
    )


def draw_rotated_text(
    base_img,
    text,
    center,
    radius,
    angle,
    font,
    fill
):
    """
    Draw individual characters around a circular path.
    """

    draw = ImageDraw.Draw(base_img)

    chars = list(text)

    # Approximate angular spacing
    spacing = 6

    start_angle = angle - ((len(chars) - 1) * spacing / 2)

    for i, char in enumerate(chars):

        current_angle = start_angle + i * spacing

        rad = math.radians(current_angle)

        x = center[0] + radius * math.sin(rad)
        y = center[1] - radius * math.cos(rad)

        # Individual transparent layer
        layer = Image.new("RGBA", (150, 150), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)

        bbox = layer_draw.textbbox(
            (0, 0),
            char,
            font=font
        )

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        layer_draw.text(
            (
                75 - w / 2,
                75 - h / 2
            ),
            char,
            font=font,
            fill=fill
        )

        # Rotate character to follow circle
        rotated = layer.rotate(
            -current_angle,
            resample=Image.Resampling.BICUBIC,
            expand=False
        )

        base_img.alpha_composite(
            rotated,
            (
                int(x - 75),
                int(y - 75)
            )
        )


def create_smart_kishan_logo(
    output_filename="smart_kishan_professional_badge.png"
):

    W = 1000
    H = 1000

    # ========================================================
    # CANVAS
    # ========================================================

    img = Image.new(
        "RGBA",
        (W, H),
        (250, 250, 247, 255)
    )

    draw = ImageDraw.Draw(img)

    cx = 500
    cy = 500

    # ========================================================
    # COLORS
    # ========================================================

    DARK_GREEN = (7, 79, 38, 255)
    GREEN = (20, 115, 48, 255)
    LIGHT_GREEN = (75, 170, 45, 255)

    GOLD = (255, 205, 30, 255)
    YELLOW = (255, 185, 20, 255)

    WHITE = (255, 255, 255, 255)

    SKY = (190, 225, 242, 255)

    FIELD = (82, 165, 48, 255)

    # ========================================================
    # 1. OUTER GOLD SHADOW
    # ========================================================

    draw.ellipse(
        [
            cx - 410,
            cy - 410,
            cx + 410,
            cy + 410
        ],
        fill=(180, 145, 20, 80)
    )

    # ========================================================
    # 2. MAIN DARK GREEN BADGE
    # ========================================================

    draw.ellipse(
        [
            cx - 395,
            cy - 395,
            cx + 395,
            cy + 395
        ],
        fill=DARK_GREEN
    )

    # ========================================================
    # 3. GOLD OUTER BORDER
    # ========================================================

    draw.ellipse(
        [
            cx - 375,
            cy - 375,
            cx + 375,
            cy + 375
        ],
        outline=GOLD,
        width=10
    )

    # ========================================================
    # 4. WHITE INNER BORDER
    # ========================================================

    draw.ellipse(
        [
            cx - 345,
            cy - 345,
            cx + 345,
            cy + 345
        ],
        outline=WHITE,
        width=12
    )

    # ========================================================
    # 5. AGRICULTURAL LANDSCAPE
    # ========================================================

    # Landscape circle
    draw.ellipse(
        [
            cx - 325,
            cy - 325,
            cx + 325,
            cy + 325
        ],
        fill=SKY
    )

    # ========================================================
    # CLOUDS
    # ========================================================

    cloud = (245, 249, 240, 255)

    clouds = [
        (250, 330, 340, 385),
        (300, 305, 400, 385),
        (360, 335, 440, 385),

        (650, 330, 735, 385),
        (700, 305, 805, 385),
        (770, 335, 850, 385)
    ]

    for box in clouds:
        draw.ellipse(box, fill=cloud)

    # ========================================================
    # 6. SUN
    # ========================================================

    sun_x = 500
    sun_y = 430
    sun_radius = 75

    # Sun rays
    for angle in range(0, 360, 30):

        rad = math.radians(angle)

        x1 = sun_x + math.cos(rad) * 95
        y1 = sun_y + math.sin(rad) * 95

        x2 = sun_x + math.cos(rad) * 135
        y2 = sun_y + math.sin(rad) * 135

        draw.line(
            [(x1, y1), (x2, y2)],
            fill=YELLOW,
            width=10
        )

    draw.ellipse(
        [
            sun_x - sun_radius,
            sun_y - sun_radius,
            sun_x + sun_radius,
            sun_y + sun_radius
        ],
        fill=(255, 193, 30, 255)
    )

    # ========================================================
    # 7. BACK HILLS
    # ========================================================

    draw.polygon(
        [
            (175, 520),
            (290, 460),
            (390, 500),
            (500, 445),
            (610, 500),
            (720, 455),
            (825, 510),
            (825, 610),
            (175, 610)
        ],
        fill=(125, 195, 105, 255)
    )

    # Front hill
    draw.polygon(
        [
            (175, 545),
            (300, 495),
            (430, 535),
            (560, 480),
            (680, 530),
            (825, 495),
            (825, 620),
            (175, 620)
        ],
        fill=(60, 145, 65, 255)
    )

    # ========================================================
    # 8. TREES
    # ========================================================

    def tree(x, y, scale=1):

        trunk_color = (75, 85, 45, 255)
        leaf_color = (35, 125, 55, 255)

        draw.rectangle(
            [
                x - int(7 * scale),
                y,
                x + int(7 * scale),
                y + int(60 * scale)
            ],
            fill=trunk_color
        )

        draw.ellipse(
            [
                x - int(35 * scale),
                y - int(80 * scale),
                x + int(35 * scale),
                y + int(10 * scale)
            ],
            fill=leaf_color
        )

    tree(700, 480, 0.9)
    tree(760, 490, 1.1)
    tree(820, 475, 0.85)

    # ========================================================
    # 9. AGRICULTURAL FIELD
    # ========================================================

    draw.polygon(
        [
            (175, 575),
            (825, 575),
            (880, 760),
            (120, 760)
        ],
        fill=FIELD
    )

    # Field rows
    for x in range(220, 850, 80):

        draw.line(
            [
                (x, 580),
                (500 + (x - 500) * 1.3, 760)
            ],
            fill=(235, 247, 215, 255),
            width=9
        )

    # ========================================================
    # 10. TRACTOR
    # ========================================================

    tractor_green = (12, 102, 52, 255)
    tractor_dark = (8, 70, 37, 255)

    # Body
    draw.rounded_rectangle(
        [285, 515, 435, 600],
        radius=12,
        fill=tractor_green
    )

    # Bonnet
    draw.rectangle(
        [415, 535, 485, 585],
        fill=tractor_green
    )

    # Cabin
    draw.line(
        [(305, 520), (305, 460)],
        fill=tractor_dark,
        width=12
    )

    draw.line(
        [(305, 460), (375, 460)],
        fill=tractor_dark,
        width=12
    )

    draw.line(
        [(375, 460), (395, 520)],
        fill=tractor_dark,
        width=12
    )

    # Cabin roof
    draw.rectangle(
        [295, 450, 385, 470],
        fill=tractor_green
    )

    # Cabin glass
    draw.polygon(
        [
            (320, 475),
            (365, 475),
            (380, 515),
            (320, 515)
        ],
        fill=(205, 230, 220, 255)
    )

    # Rear wheel
    draw.ellipse(
        [260, 550, 340, 630],
        fill=tractor_dark
    )

    draw.ellipse(
        [280, 570, 320, 610],
        fill=WHITE
    )

    # Front wheel
    draw.ellipse(
        [425, 555, 475, 605],
        fill=tractor_dark
    )

    draw.ellipse(
        [438, 568, 462, 592],
        fill=WHITE
    )

    # Exhaust
    draw.rectangle(
        [455, 480, 465, 535],
        fill=tractor_dark
    )

    # ========================================================
    # 11. LARGE LEAF CANOPY
    # ========================================================

    # Main upper leaf
    draw.polygon(
        [
            (235, 300),
            (330, 205),
            (475, 155),
            (650, 175),
            (550, 235),
            (420, 285),
            (300, 305)
        ],
        fill=(18, 115, 48, 255)
    )

    # Leaf vein
    draw.line(
        [
            (255, 285),
            (390, 220),
            (550, 180)
        ],
        fill=WHITE,
        width=9
    )

    # Second leaf
    draw.polygon(
        [
            (540, 235),
            (650, 180),
            (775, 235),
            (690, 295),
            (580, 305)
        ],
        fill=(75, 165, 42, 255)
    )

    # ========================================================
    # 12. TOP "GOVT COMPLIANT"
    # ========================================================

    top_font = get_font(42, True)

    # Use curved text
    draw_rotated_text(
        img,
        "GOVT COMPLIANT",
        (500, 500),
        340,
        0,
        top_font,
        WHITE
    )

    # ========================================================
    # 13. SMALL LEAF ICONS
    # ========================================================

    def small_leaf(x, y, flip=False):

        if not flip:

            draw.polygon(
                [
                    (x, y + 40),
                    (x - 25, y),
                    (x + 5, y - 30),
                    (x + 25, y + 10)
                ],
                fill=(120, 205, 35, 255)
            )

        else:

            draw.polygon(
                [
                    (x, y + 40),
                    (x + 25, y),
                    (x - 5, y - 30),
                    (x - 25, y + 10)
                ],
                fill=(120, 205, 35, 255)
            )

    small_leaf(220, 285)
    small_leaf(780, 285, True)

    # ========================================================
    # 14. SMART KISHAN RIBBON
    # ========================================================

    ribbon_y1 = 555
    ribbon_y2 = 735

    # Ribbon shadow
    draw.polygon(
        [
            (40, 620),
            (110, 580),
            (890, 580),
            (960, 620),
            (925, 735),
            (75, 735)
        ],
        fill=(5, 65, 32, 255)
    )

    # Main ribbon
    draw.polygon(
        [
            (70, 585),
            (140, 550),
            (860, 550),
            (930, 585),
            (900, 710),
            (100, 710)
        ],
        fill=DARK_GREEN
    )

    # Gold ribbon border
    draw.line(
        [
            (70, 585),
            (140, 550),
            (860, 550),
            (930, 585)
        ],
        fill=GOLD,
        width=10
    )

    draw.line(
        [
            (100, 710),
            (900, 710)
        ],
        fill=GOLD,
        width=10
    )

    # Ribbon side tails
    draw.polygon(
        [
            (70, 600),
            (30, 625),
            (75, 700),
            (110, 690)
        ],
        fill=GREEN
    )

    draw.polygon(
        [
            (930, 600),
            (970, 625),
            (925, 700),
            (890, 690)
        ],
        fill=GREEN
    )

    # ========================================================
    # 15. SMART KISHAN TEXT
    # ========================================================

    smart_font = get_font(76, True)

    smart_text = "SMART"
    kishan_text = "KISHAN"

    # Measure
    smart_box = draw.textbbox(
        (0, 0),
        smart_text,
        font=smart_font
    )

    kishan_box = draw.textbbox(
        (0, 0),
        kishan_text,
        font=smart_font
    )

    smart_w = smart_box[2] - smart_box[0]
    kishan_w = kishan_box[2] - kishan_box[0]

    total_w = smart_w + kishan_w + 25

    start_x = (1000 - total_w) / 2

    # SMART
    draw.text(
        (start_x, 585),
        smart_text,
        font=smart_font,
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 60, 30, 255)
    )

    # KISHAN
    draw.text(
        (start_x + smart_w + 25, 585),
        kishan_text,
        font=smart_font,
        fill=GOLD,
        stroke_width=2,
        stroke_fill=(0, 60, 30, 255)
    )

    # ========================================================
    # 16. 4R CERTIFIED SECTION
    # ========================================================

    # Rounded certification box
    draw.rounded_rectangle(
        [265, 760, 735, 850],
        radius=35,
        fill=DARK_GREEN,
        outline=GOLD,
        width=6
    )

    cert_font = get_font(43, True)

    cert_text = "4R CERTIFIED"

    cert_box = draw.textbbox(
        (0, 0),
        cert_text,
        font=cert_font
    )

    cert_w = cert_box[2] - cert_box[0]

    draw.text(
        ((1000 - cert_w) / 2, 780),
        cert_text,
        font=cert_font,
        fill=WHITE
    )

    # ========================================================
    # 17. STARS
    # ========================================================

    def star(cx, cy, outer, inner, color):

        points = []

        for i in range(10):

            angle = math.radians(-90 + i * 36)

            r = outer if i % 2 == 0 else inner

            points.append(
                (
                    cx + math.cos(angle) * r,
                    cy + math.sin(angle) * r
                )
            )

        draw.polygon(
            points,
            fill=color
        )

    star(225, 805, 25, 10, GOLD)
    star(775, 805, 25, 10, GOLD)

    # ========================================================
    # 18. BOTTOM LEAF ICON
    # ========================================================

    draw.polygon(
        [
            (500, 930),
            (455, 875),
            (465, 835),
            (500, 875)
        ],
        fill=(110, 195, 35, 255)
    )

    draw.polygon(
        [
            (500, 930),
            (545, 875),
            (535, 835),
            (500, 875)
        ],
        fill=(110, 195, 35, 255)
    )

    # Central stem
    draw.line(
        [(500, 935), (500, 855)],
        fill=(80, 150, 30, 255),
        width=7
    )

    # ========================================================
    # 19. FINAL OUTER RING
    # ========================================================

    draw.ellipse(
        [
            cx - 395,
            cy - 395,
            cx + 395,
            cy + 395
        ],
        outline=DARK_GREEN,
        width=14
    )

    draw.ellipse(
        [
            cx - 380,
            cy - 380,
            cx + 380,
            cy + 380
        ],
        outline=GOLD,
        width=5
    )

    # ========================================================
    # 20. SAVE
    # ========================================================

    img.save(
        output_filename,
        "PNG",
        optimize=True
    )

    print("==========================================")
    print(" SMART KISHAN LOGO GENERATED SUCCESSFULLY")
    print("==========================================")
    print(f"File: {output_filename}")
    print("Resolution: 1000 x 1000")
    print("Format: PNG")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_smart_kishan_logo(
        "smart_kishan_professional_badge.png"
    )
