from google import genai
from google.genai import types
from PIL import Image
import io
import base64
import os
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
from retrieve import hierarchical_retrieval
from utils import FIG_RAG_DIR, save_image_to_file, save_prompt_to_file
from invoke import prompt_writer

from config import Config
config = Config()
model_name = config.generation_model
resolution = "1K"

NAME = "Code2MCP"
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENERATE = True

def call_image_model(prompt_sequence):
    """
    Returns:
        tuple: (image_bytes, input_token_count, generated_pixel_count)
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_sequence,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                temperature=0.0,
                image_config=types.ImageConfig(
                    image_size=resolution
                ),
            )
        )
        
        # 1. Track Input Tokens
        input_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count

        for part in response.parts:
            if part.inline_data:
                img_data = part.inline_data.data
                
                # 2. Track Generated Pixels
                try:
                    with Image.open(io.BytesIO(img_data)) as img:
                        width, height = img.size
                        generated_pixels = width * height
                except Exception as e:
                    print(f"Warning: Could not calculate pixels: {e}")
                    generated_pixels = 0
                
                return img_data, input_tokens, generated_pixels
                
        return None, 0, 0
    except Exception as e:
        print(f"Error calling model: {e}")
        return None, 0, 0

def image_to_base64_data_url(pil_image):
    buffered = io.BytesIO()
    # Convert RGB to ensure compatibility (remove alpha channel if problematic for some encoders)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def description(title, abstract, caption, prompt_writer, retrieved_docs=None):
    system_prompt = (
        "You are an expert on translating academic writing to visual specification. "
        "Your job is to describe what you think the diagram should look like. "
        "You will be given the title and abstract of a scientific paper, along with the intended caption of a new diagram. "
        "You will also be given a set of reference diagrams and their captions. "
        "Study how the references visualize their captions through visual elements."
        "\n\n"
        "Some requirements on your specification:\n"
        "- Ensure the final description is self-contained.\n"
        "- Detailedly state the texts in every component.\n"
        "- Reference the visual traits in the reference diagrams (e.g., 'Use the same pastel blue shading as Reference 1', 'Copy the dotted arrow style from Reference 2').\n" # RAG
        "- Use your internal knowledge as well as the reference diagrams to specify the layout, color scheme, and icons.\n" # RAG
        "- If you don't plan to use a reference diagram, you don't need to say anything about it."
        "\n\n"
        "Final note:\n"
        "Make sure your specification match the intended caption of the new diagram."
    )

    # Build Multimodal Content List
    user_prompt_content = []
    
    # 1. Add Introductory Text
    user_prompt_content.append({
        "type": "text", 
        "text": "Here are the reference diagrams to study for style:"
    })

    # 2. Add Images (Converted to Base64 Dicts)
    for i, doc in enumerate(retrieved_docs):
        try:
            metadata = doc.metadata
            path = os.path.join(FIG_RAG_DIR, metadata['figure_path'])
            if os.path.exists(path):
                img = Image.open(path)
                
                # --- CONVERSION HAPPENS HERE ---
                img_data = image_to_base64_data_url(img)
                
                user_prompt_content.append({
                    "type": "text", 
                    "text": f"\n--- Reference {i+1} ---"
                })
                user_prompt_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_data}
                })
                user_prompt_content.append({
                    "type": "text",
                    "text": (
                        f"<reference_caption>{metadata['figure_caption']}</reference_caption>"
                    )
                })
            else:
                print(f"Warning: RAG Image missing at {path}")
        except Exception as e:
            print(f"Skipping corrupt reference image: {e}")

    # 3. Add Final Instruction Text
    user_prompt_content.append({
        "type": "text",
        "text": (
            "Describe a diagram for:\n"
            f"<title>{title}</title>\n"
            f"<abstract>{abstract}</abstract>\n"
            f"<caption>{caption}</caption>\n"
            "Begin your response with '## Diagram Specification'"
        )
    })

    return prompt_writer.invoke(user_prompt=user_prompt_content, system_prompt=system_prompt)


def build_rag_prompt(description, retrieved_docs):
    # This function remains mostly the same, serving to assemble the final payload 
    # for the Image Generator (Nano Banana).
    prompt_sequence = []
    prompt_sequence.append(description)
    
    if retrieved_docs:
        reference_prompt = (
            "\n## Reference Diagrams\n"
        )
        prompt_sequence.append(reference_prompt)

        for i, doc in enumerate(retrieved_docs):
            try:
                path = os.path.join(FIG_RAG_DIR, doc.metadata['figure_path'])
                if os.path.exists(path):
                    img = Image.open(path)
                    prompt_sequence.append(f"\n--- Reference {i+1} ---\n")
                    prompt_sequence.append(img)
            except Exception:
                continue
                
    return prompt_sequence

def run_experiment(title, abstract, caption):
    print(f"=== Experiment: {title} ===")
    print("\n--- Running Condition: WITH RAG ---")
    docs = hierarchical_retrieval(
        query_title=title, 
        query_abstract=abstract, 
        query_caption=caption, 
        k=3
    )

    if docs:
        print(f"Retrieved {len(docs)} reference figures.")
        
        # CRITICAL CHANGE: We now pass docs INTO the description generator
        # This allows the prompt_writer to see the images and adapt the text specification
        spec_rag = description(title, abstract, caption, prompt_writer, retrieved_docs=docs)
        
        # Then we assume the standard build process to bundle it for the image model
        prompt_rag = build_rag_prompt(spec_rag, docs)
        
        save_prompt_to_file(prompt_rag, os.path.join(OUTPUT_DIR, f"{NAME}_WITH_RAG_prompt.txt"))
        if GENERATE:
            img_bytes, tokens, pixels = call_image_model(prompt_rag)
            save_image_to_file(img_bytes, os.path.join(OUTPUT_DIR, f"{NAME}_WITH_RAG.png"))
    else:
        print("No documents retrieved! Exiting...")

if __name__ == "__main__":
    with open("title.txt", 'r') as f:
        title = f.read()
    with open("abstract.txt", 'r') as f:
        abstract = f.read()
    with open("caption.txt", 'r') as f:
        caption = f.read()
    
    run_experiment(title, abstract, caption)