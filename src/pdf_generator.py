from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from PIL import Image
import io


def generate_print_sheet(passport_image: Image.Image) -> io.BytesIO:
    """
    Generates a 4x6 inch PDF sheet with 6 passport photos in a 3x2 grid.
    Includes crop marks for cutting.
    """
    pdf_buffer = io.BytesIO()

    # 4x6 inches in points (1 inch = 72 points)
    page_width = 4 * inch
    page_height = 6 * inch

    c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

    # Passport photo size: 35x45mm = approx 1.378 x 1.772 inches
    photo_w = 1.378 * inch
    photo_h = 1.772 * inch

    # Grid layout: 2 columns, 3 rows
    cols, rows = 2, 3

    # Calculate spacing to center the grid
    x_margin = (page_width - (cols * photo_w)) / (cols + 1)
    y_margin = (page_height - (rows * photo_h)) / (rows + 1)

    # Convert PIL Image to ImageReader for ReportLab
    img_buffer = io.BytesIO()
    passport_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    for row in range(rows):
        for col in range(cols):
            x = x_margin + col * (photo_w + x_margin)
            y = page_height - (y_margin + (row + 1) * photo_h + row * y_margin)

            # Draw the photo
            c.drawImage(img_reader, x, y, width=photo_w, height=photo_h)

            # Draw crop marks (corner brackets)
            mark_len = 0.1 * inch
            c.setStrokeColorRGB(0, 0, 0)
            c.setLineWidth(0.5)

            # Top-left
            c.line(x, y + photo_h, x + mark_len, y + photo_h)
            c.line(x, y + photo_h, x, y + photo_h - mark_len)
            # Top-right
            c.line(x + photo_w, y + photo_h, x + photo_w - mark_len, y + photo_h)
            c.line(x + photo_w, y + photo_h, x + photo_w, y + photo_h - mark_len)
            # Bottom-left
            c.line(x, y, x + mark_len, y)
            c.line(x, y, x, y + mark_len)
            # Bottom-right
            c.line(x + photo_w, y, x + photo_w - mark_len, y)
            c.line(x + photo_w, y, x + photo_w, y + mark_len)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer
