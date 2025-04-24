from openai import OpenAI
import base64

client = OpenAI()

result = client.images.edit(
    model="gpt-image-1",
    image=[
      open("fotos/estrela.png", "rb"),
    
    ],
    prompt="Generate a photorealistic image of a gift basket on a white background labeled 'Relax & Unwind' with a ribbon and handwriting-like font, containing all the items in the reference pictures"
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("gift-basket.png", "wb") as f:
    f.write(image_bytes)