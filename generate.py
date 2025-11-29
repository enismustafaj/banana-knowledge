from pathlib import Path
from typing import Dict, Any, List

from google import genai
from google.genai import types


IMAGE_MODEL_ID = "gemini-3-pro-image-preview"  # Nano Banana Pro (image)

def generate_workflow_diagram(
    client: genai.Client,
    workflow: Dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Use Nano Banana Pro (Gemini 3 Pro Image) to create a simple workflow diagram
    (boxes + arrows) from the step texts.
    """

    workflow_name: str = workflow.get("workflow_name", "Workflow")
    steps: List[Dict[str, Any]] = workflow.get("steps", [])

    step_lines = []
    for step in steps:
        sid = step.get("id")
        text = step.get("text", "").strip()
        if sid is None or not text:
            continue
        # Shorten very long texts for the diagram label
        short_text = text
        # if len(short_text) > 120:
        #     short_text = short_text[:117] + "..."
        step_lines.append(f"- Step {sid + 1}: {short_text}")

    steps_block = "\n".join(step_lines) if step_lines else "- (No steps extracted)"

    diagram_prompt = f"""
You are designing a modern cloud-incident workflow diagram in a style inspired by
cutting-edge cloud security tools (think Wiz-like dashboards and node graphs).

Goal:
Create a visually striking, Wiz-style flow diagram for this workflow:

Workflow name: "{workflow_name}"

Content:
- One node/box per step.
- Arrows between nodes in strict numerical order (1 → 2 → 3 → ...).
- Each node label is a short summary of the step.
- Include a START node before step 1 and an END node after the last step.

Steps:
{steps_block}

Visual style (Wiz-inspired):
- Dark, cyber-style background with a subtle gradient (deep navy / midnight purple).
- Neon accent colors (teal, cyan, electric blue, lime) for nodes and connecting lines.
- Nodes can look like floating cards or bubbles with soft glow and depth.
- Thin, glowing connector lines, optionally curved, forming a clean graph-like flow.
- Light grid or subtle geometric pattern in the background to evoke a cloud/security dashboard.
- Use a modern, techy font; legible but slightly futuristic.
- Optional soft halos or outlines around critical steps to draw attention.

Layout:
- Left-to-right primary flow, with a gentle downward progression.
- 16:9 aspect ratio.
- Balanced spacing and generous padding so it works as a hero image in docs or slides.

Avoid:
- Real company logos or explicit vendor names.
- Dense paragraphs inside nodes; 
""".strip()
    
    print(steps_block)

    response = client.models.generate_content(
        contents=diagram_prompt,
        model="gemini-3-pro-image-preview",
        config=types.GenerateContentConfig(
            response_modalities=["Image"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
            ),
        ),
    )

    # Save the first image returned
    for part in response.parts:
        img = part.as_image()
        if img:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
            return output_path

    raise RuntimeError("No image returned from image model.")
