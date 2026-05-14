#!/usr/bin/env python3
"""Map document content to image assets"""
import json, os, glob, sys

IMAGE_BASE = "/tmp/batch50_output"
SERVER_BASE = "http://172.24.59.194:8098"

def build_full_map():
    """Build comprehensive mapping of all config codes and their images"""
    all_codes = {}
    all_steps = {}
    all_faq = {}
    
    for d in sorted(os.listdir(IMAGE_BASE)):
        dpath = os.path.join(IMAGE_BASE, d)
        images_dir = os.path.join(dpath, "images")
        if not os.path.isdir(dpath) or not os.path.isdir(images_dir):
            continue
        
        json_files = glob.glob(os.path.join(dpath, "*_structured.json"))
        if not json_files:
            continue
        
        with open(json_files[0]) as f:
            data = json.load(f)
        
        # Config codes
        for cc in data.get("config_codes", []):
            img_ref = cc.get("image_ref", "")
            name = cc.get("code_name", "")
            desc = cc.get("description", "")
            product = ""
            cls = data.get("classifiers", {})
            if isinstance(cls, dict):
                product = f"{cls.get('product_name','')} {cls.get('model','')}".strip()
            if img_ref:
                url = f"{SERVER_BASE}/{d}/images/{os.path.basename(img_ref)}"
                all_codes[name] = {
                    "image_url": url,
                    "description": desc,
                    "product": product
                }
        
        # Operation steps
        for step in data.get("operation_steps", []):
            img_refs = step.get("image_refs", [])
            if img_refs:
                key = f"步骤{step.get('step_number','')}-{step.get('title','')}"
                all_steps[key] = {
                    "image_urls": [f"{SERVER_BASE}/{d}/images/{os.path.basename(r)}" for r in img_refs],
                    "description": step.get("description", "")
                }
    
    return all_codes, all_steps

def match_images(query: str, all_codes: dict, top_k: int = 5) -> dict:
    """Match config codes by query keywords"""
    q = query.lower()
    keywords = q.replace("配置码", "").replace("码", "").split()
    
    results = {}
    for name, info in all_codes.items():
        score = 0
        name_lower = name.lower()
        desc_lower = info["description"].lower()
        
        for kw in keywords:
            if kw in name_lower:
                score += 2
            if kw in desc_lower:
                score += 1
            # Match product info too
            if kw in info.get("product", "").lower():
                score += 1
        
        if score > 0:
            results[name] = {**info, "match_score": score}
    
    # Order by match score
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["match_score"], reverse=True)[:top_k])
    return sorted_results

# Build once at module level
_ALL_CODES, _ALL_STEPS = build_full_map()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        results = match_images(query, _ALL_CODES)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "total_config_codes": len(_ALL_CODES),
            "total_steps": len(_ALL_STEPS)
        }, ensure_ascii=False, indent=2))
