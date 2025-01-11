from PIL import Image
import base64
from io import BytesIO

from .schema import BoundingBox, Polygon


def base64_encode(image: Image.Image, format="PNG", url_prefix=False):
  """
  Encodes a PIL Image to a base64 string.

  Args:
      image (PIL.Image.Image): The image to encode.
      format (str): The format to save the image as (e.g., "PNG", "JPEG").
      url_prefix (bool): If True, adds a URL prefix to the base64 string.

  Returns:
      str: The base64-encoded string of the image, optionally with a URL prefix.
  """
  buffered = BytesIO()
  image.save(buffered, format=format)
  buffered.seek(0)
  img_base64 = base64.b64encode(buffered.read()).decode('utf-8')
  if url_prefix:
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"
  return img_base64


def base64_decode(text):
  if text.startswith("data:"):
    text = text.split(",", 1)[1]
  img_data = base64.b64decode(text)
  buffered = BytesIO(img_data)
  img = Image.open(buffered)
  return img




def draw_boxes(image: Image.Image, boxes: list[BoundingBox | dict]):
  from PIL import ImageDraw, ImageFont
  draw = ImageDraw.Draw(image)

  try:
    font = ImageFont.truetype("arial.ttf", size=16)
  except IOError:
    font = ImageFont.load_default()

  for box in boxes:
    if isinstance(box, dict):
      box = BoundingBox(**box)
    draw.rectangle(box.xyxy, outline="red", width=2)

    x1, y1, x2, y2 = box.xyxy
    # Calculate text size to create a background box
    text_bbox = font.getbbox(box.label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_bg_box = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]

    # Draw a filled rectangle behind the text for better visibility
    draw.rectangle(text_bg_box, fill="red")

    # Draw the text label
    draw.text((x1 + 2, y1 - text_height - 2), box.label, fill="white", font=font)
  return image



def draw_polys(image: Image.Image, polys: list[Polygon | dict]):
  from PIL import ImageDraw
  draw = ImageDraw.Draw(image)

  # Draw each quad
  for poly in polys:
    if isinstance(poly, dict):
      poly = Polygon(**poly)
    draw.polygon(
      list(zip(poly.points[::2], poly.points[1::2])), outline="blue", width=3
    )
  return image