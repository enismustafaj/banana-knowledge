import os
import json
from pathlib import Path
from typing import Dict, Any, List

from google import genai
from google.genai import types

# ---------- CONFIG ----------

# Where your partner stores the extracted text from the screen recording
TRANSCRIPT_PATH = Path("data/transcript.txt")

# Where we’ll save outputs
WORKFLOW_JSON_PATH = Path("data/workflow.json")
DIAGRAM_IMAGE_PATH = Path("data/workflow_diagram.png")

# Gemini models
TEXT_MODEL_ID = "gemini-3-pro-preview"         # for text/workflow extraction
IMAGE_MODEL_ID = "gemini-3-pro-image-preview"  # Nano Banana Pro (image)


# ---------- HELPERS ----------

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def read_transcript(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found at: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Transcript file {path} is empty.")
    return text


def generate_workflow(client: genai.Client, transcript: str) -> Dict[str, Any]:
    """
    Use Gemini 3 to turn a narration into a very simple workflow schema:

    {
      "workflow_name": "string",
      "steps": [
        { "id": 1, "text": "..." }
      ]
    }
    """

    system_instructions = """
You will receive a transcript of someone walking through a process or workflow.

Your task:
1. Infer a concise WORKFLOW NAME.
2. Break the narration into a sequence of clear, ordered steps.

Use this exact JSON schema, and return ONLY valid JSON (no extra text):

{
  "workflow_name": "string",
  "steps": [
    {
      "id": 1,
      "text": "string"
    }
  ]
}

Rules:
- "workflow_name" should be a short noun phrase, e.g. "Customer Refund Workflow".
- "steps" must be an array.
- "id" must be integers starting at 1 and increasing by 1 (1, 2, 3, ...).
- "text" should be the extracted / cleaned step text, 1–3 sentences that describe what happens in that step.
- Focus on the PROCESS, not implementation details or code.
""".strip()

    prompt = f"""{system_instructions}

TRANSCRIPT:
\"\"\"{transcript}\"\"\""""

    print(f"🔍 DEBUG: About to call API with model: {TEXT_MODEL_ID}")
    print(f"🔍 DEBUG: Prompt length: {len(prompt)} characters")
    print(f"🔍 DEBUG: Transcript length: {len(transcript)} characters")
    
    response = client.models.generate_content(
        model=TEXT_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2048,
        ),
    )

    print("✅ DEBUG: API call successful, received response")
    raw_text = response.text
    print(f"🔍 DEBUG: Response length: {len(raw_text)} characters")
    print(f"🔍 DEBUG: First 200 chars of response: {raw_text[:200]}")

    # Try to parse as pure JSON
    try:
        print("🔍 DEBUG: Attempting to parse as JSON...")
        data = json.loads(raw_text)
        print("✅ DEBUG: Successfully parsed JSON directly")
    except json.JSONDecodeError as e:
        print(f"⚠️  DEBUG: JSON decode failed: {e}")
        # If the model adds extra text, try to cut out the JSON block
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        print(f"🔍 DEBUG: Trying to extract JSON from position {start} to {end}")
        if start == -1 or end == -1:
            print(f"❌ DEBUG: Could not find JSON braces. Full response:\n{raw_text}")
            raise RuntimeError("Could not find JSON object in model output.")
        data = json.loads(raw_text[start:end + 1])
        print("✅ DEBUG: Successfully extracted and parsed JSON")

    # Optional: light sanity check
    print(f"🔍 DEBUG: Validating workflow data...")
    if "workflow_name" not in data or "steps" not in data:
        print(f"❌ DEBUG: Missing keys. Data keys: {list(data.keys())}")
        raise RuntimeError("Model output missing required keys 'workflow_name' or 'steps'.")
    
    print(f"✅ DEBUG: Validation passed. Workflow name: {data.get('workflow_name')}")
    print(f"✅ DEBUG: Number of steps: {len(data.get('steps', []))}")
    
    return data


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
        if len(short_text) > 120:
            short_text = short_text[:117] + "..."
        step_lines.append(f"- Step {sid}: {short_text}")

    steps_block = "\n".join(step_lines) if step_lines else "- (No steps extracted)"

    diagram_prompt = f"""
Create a clear, minimal flowchart-style workflow diagram.

Workflow name: "{workflow_name}"

Show:
- One box per step
- Arrows between boxes in numerical order (1 -> 2 -> 3 ...)
- Each box label should reflect the meaning of the step.

Steps:
{steps_block}

Design:
- Light background, dark text
- Simple shapes and arrows
- Left-to-right or top-to-bottom flow
- 16:9 aspect ratio
- Suitable as a visual summary of the workflow
""".strip()

    response = client.models.generate_content(
        model=IMAGE_MODEL_ID,
        contents=diagram_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["Image"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="1K",
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


# ---------- MAIN SCRIPT ----------

def main():
    print("🔧 Reading transcript for workflow...")
    transcript = read_transcript(TRANSCRIPT_PATH)

    print("🤖 Initializing Gemini client...")
    client = get_client()

    print("📄 Generating simple WORKFLOW JSON with Gemini 3...")
    workflow = generate_workflow(client, transcript)

    WORKFLOW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKFLOW_JSON_PATH.write_text(
        json.dumps(workflow, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✅ Workflow JSON saved to {WORKFLOW_JSON_PATH}")

    print("🖼  Generating workflow diagram with Nano Banana Pro...")
    diagram_path = generate_workflow_diagram(client, workflow, DIAGRAM_IMAGE_PATH)
    print(f"✅ Workflow diagram saved to {diagram_path}")

    print("\nDone!")
    print("Your teammate can now:")
    print(f"- Read workflow structure from: {WORKFLOW_JSON_PATH}")
    print(f"- Display the diagram image from: {diagram_path}")


if __name__ == "__main__":
    main()
