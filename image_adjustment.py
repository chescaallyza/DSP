from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

def apply_style(image, style):
    if style == "None":
        return image
    elif style == "Sketch":
        return image.convert("L").filter(ImageFilter.FIND_EDGES)
    elif style == "Black and White":
        return image.convert("L").convert("RGB")
    elif style == "Sepia":
        sepia_image = ImageOps.colorize(
            image.convert("L"),
            black=(80, 60, 40),
            white=(220, 200, 180),
        )
        contrast_enhancer = ImageEnhance.Contrast(sepia_image)
        return contrast_enhancer.enhance(1.1)
    elif style == "Vintage":
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (215, 175, 130, 100))
        return Image.alpha_composite(image, overlay).convert("RGB")
    return image

def adjust_image(image, brightness, contrast, saturation, vignette, warm):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1 + (brightness / 100))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + (contrast / 100))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1 + (saturation / 100))

    if vignette != 0:
        vignette_mask = Image.new("L", image.size, 255)
        draw = ImageDraw.Draw(vignette_mask)
        width, height = image.size
        draw.ellipse(
            [(width * 0.2, height * 0.2), (width * 0.8, height * 0.8)],
            fill=max(0, 255 - int(abs(vignette) * 2.55)),
        )
        vignette_mask = vignette_mask.filter(ImageFilter.GaussianBlur(50))
        vignette_overlay = Image.new("RGB", image.size, (0, 0, 0))
        image = Image.composite(vignette_overlay, image, vignette_mask)

    return image
