from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("Codatta/MM-Food-100K", split="train")

data = []
for item in ds:
    # Combine some fields into tags
    tags = []
    if item.get("food_type"):
        tags.append(item["food_type"])
    if item.get("ingredients"):
        tags.extend([i.strip() for i in item["ingredients"].split(",")])
    if item.get("cooking_method"):
        tags.append(item["cooking_method"])
    
    data.append({
        "name": item.get("dish_name") or "Unknown",
        "tags": tags,
        "imageUrl": item.get("image_url") or None
    })

# Save JSON
with open("food_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
