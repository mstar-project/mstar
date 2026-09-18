"""Text-to-image, then an edit of the result, via the Python SDK.

Start a server for an image model that serves edits first, e.g.:  mstar serve bagel
"""

from mstar import MStarClient

client = MStarClient("http://localhost:8000")

png = client.generate_image("a cat holding a sign that says hello world", seed=0)
with open("out.png", "wb") as f:
    f.write(png)
print(f"wrote out.png - {len(png)} bytes")

edited = client.edit_image("make the sign say goodbye, watercolor style", "out.png", seed=1)
with open("edit.png", "wb") as f:
    f.write(edited)
print(f"wrote edit.png - {len(edited)} bytes")
