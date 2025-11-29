from google import genai
from google.genai import types
import os
from database import r
import json 

client = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=[
        "You are an assitants that understands and records the workflow steps that a person performs.",
        "You must analyze the images and extract the information regarding the action that is performed!"
    ]
)

for filename in os.listdir("assets/workflow"):

    i = 0

    with open(filename, 'rb') as f:

        image_bytes = f.read()
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
                ),
                'Analyze and provide a summary of an action being performed!'
            ]
        )

        step = {
            id: i,
            text: response
        }

        json_payload = json.dumps(step)

        r.rpush("workflows:workflow", json_payload)

    i = i + 1



