import duckdb
import os
import torch
from PIL import Image
from tqdm import tqdm
from utils import load_model_and_tokenizer, get_ensemble_features

FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR")
TMLR_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "TMLR")
DB_PATH = os.path.join(TMLR_DIR, "research.db")

BATCH_SIZE = 16


def main():
    model, tokenizer, preprocess = load_model_and_tokenizer()
    model.eval()

    prompts_dict = {
        "diagram": [
            "a scientific diagram or schematic",
            "a system architecture or flow chart",
            "a neural network or computational graph",
            "a block diagram with arrows",
        ],
        "plot": [
            "a statistical data chart",
            "a line chart or scatter plot with axes",
            "a bar chart or histogram",
            "a confusion matrix heatmap",
            "a quantitative visualization of results",
            "curves showing performance metrics",
        ],
        "photo": [
            "a photograph of a real object",
            "a microscopy or camera image",
            "a realistic photo without text",
            "a natural scene",
            "a screenshot of a video game or simulation",
            "a pixel art illustration or comic strip",
        ],
        "other": [
            "a full page of dense text",
            "a standalone mathematical formula",
            "a table of numbers without graphics",
            "garbage noise or corruption",
            "a blank white page",
        ],
    }

    class_map = {0: "diagram", 1: "plot", 2: "photo", 3: "other"}

    print("Encoding and ensembling prompts...")
    text_features = get_ensemble_features(model, tokenizer, prompts_dict)

    con = duckdb.connect(DB_PATH)

    query = """
        SELECT platform_id, figure_number, figure_path
        FROM Figures
        WHERE figure_type = ''
    """
    todos = con.execute(query).fetchall()
    print(f"Found {len(todos)} figures to classify.")

    updates = []
    batch_imgs = []
    batch_ids = []

    for pid, fnum, fig_path in tqdm(todos, desc="Classifying"):
        fig_path = os.path.join(FIG_RAG_DIR, fig_path)
        if not os.path.exists(fig_path):
            continue

        try:
            image = preprocess(Image.open(fig_path).convert("RGB")).unsqueeze(0)
            batch_imgs.append(image)
            batch_ids.append((pid, fnum))
        except Exception:
            continue

        if len(batch_imgs) >= BATCH_SIZE:
            with torch.no_grad():
                image_input = torch.cat(batch_imgs)
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                vals, indices = similarity.topk(1)

                for i, idx_tensor in enumerate(indices):
                    label = class_map[idx_tensor.item()]
                    confidence = vals[i].item()
                    updates.append((label, confidence, *batch_ids[i]))

            batch_imgs = []
            batch_ids = []

    if batch_imgs:
        with torch.no_grad():
            image_input = torch.cat(batch_imgs)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            vals, indices = similarity.topk(1)
            for i, idx_tensor in enumerate(indices):
                label = class_map[indices[i].item()]
                confidence = vals[i].item()
                updates.append((label, confidence, *batch_ids[i]))

    if updates:
        print(f"Updating {len(updates)} records in DuckDB...")
        con.executemany(
            """
            UPDATE Figures
            SET figure_type = ?, confidence = ?
            WHERE platform_id = ? AND figure_number = ?
            """,
            updates,
        )

    con.close()
    print("Classification complete.")


if __name__ == "__main__":
    main()
