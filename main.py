from google import genai
from google.genai import types
import os
from database import add_step, get_steps, delete_steps
from pathlib import Path

from generate import generate_workflow_diagram

DIAGRAM_IMAGE_PATH = Path("assets/data/workflow_diagram.png")

# text_client = genai.GenerativeModel(
    
# )

# image_client = genai.GenerativeModel(
#     model_name="gemini-3-pro-image-preview"
# )
client = genai.Client()
i = 0

for filename in sorted(os.listdir("assets/workflow")):

    with open(f"assets/workflow/{filename}", 'rb') as f:

        image_bytes = f.read()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
                ),
                'Analyze and provide a summary of an action being performed! Provide one single sentence as summary!'
            ]
        )
        print(response.text)

        step = {
            "id": i,
            "text": response.text
        }

        add_step(step, "workflow")

    i = i + 1

workflow = {
    "workflow_name": "workflow",
    "steps": get_steps("workflow")
}

print("\n")

print(workflow)

print("🖼  Generating workflow diagram with Nano Banana Pro...")
diagram_path = generate_workflow_diagram(client, workflow, DIAGRAM_IMAGE_PATH)
print(f"✅ Workflow diagram saved to {diagram_path}")

delete_steps("workflow")

print("\nDone!")
