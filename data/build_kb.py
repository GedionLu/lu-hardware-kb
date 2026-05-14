#!/usr/bin/env python3
"""构建配置码数据库 + 操作步骤库"""
import json, os, glob, sys, re

OUTPUT_DIR = "/home/admin/.openclaw/workspace/lu/data"
SRC_DIR = "/tmp/batch50_output"
SERVER_URL = "http://172.24.59.194:8098"

def parse_docs():
    config_codes = []
    steps = []
    
    for d in sorted(os.listdir(SRC_DIR)):
        dpath = os.path.join(SRC_DIR, d)
        if not os.path.isdir(dpath):
            continue
            
        json_files = glob.glob(os.path.join(dpath, "*_structured.json"))
        if not json_files:
            continue
        
        with open(json_files[0]) as f:
            data = json.load(f)
        
        # Get product context
        cl = data.get("classifiers", {})
        if isinstance(cl, dict):
            product_name = cl.get("product_name", "")
            model = str(cl.get("model", ""))
        else:
            product_name = ""
            model = ""
        
        images_dir = os.path.join(dpath, "images")
        
        # Extract config codes
        for cc in data.get("config_codes", []):
            img_ref = cc.get("image_ref", "")
            img_url = ""
            if img_ref and os.path.isdir(images_dir):
                img_name = os.path.basename(img_ref)
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    img_url = f"{SERVER_URL}/{d}/images/{img_name}"
            
            entry = {
                "type": "config_code",
                "code_name": cc.get("code_name", ""),
                "description": cc.get("description", ""),
                "product_name": product_name,
                "model": model,
                "image_url": img_url,
                "image_path": img_url.replace(SERVER_URL, ""),
                "source_file": d,
                "keywords": f"{cc.get('code_name','')} {cc.get('description','')} {product_name} {model} {d}"
            }
            config_codes.append(entry)
        
        # Extract operation steps
        for step in data.get("operation_steps", []):
            img_refs = step.get("image_refs", [])
            img_urls = []
            for r in img_refs:
                if os.path.isdir(images_dir):
                    img_name = os.path.basename(r)
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        img_urls.append(f"{SERVER_URL}/{d}/images/{img_name}")
            
            entry = {
                "type": "step",
                "step_number": step.get("step_number", ""),
                "title": step.get("title", ""),
                "description": step.get("description", ""),
                "warnings": step.get("warnings", []),
                "product_name": product_name,
                "model": model,
                "image_urls": img_urls,
                "source_file": d,
                "keywords": f"{step.get('title','')} {step.get('description','')} {product_name} {model} {d}"
            }
            steps.append(entry)
    
    return config_codes, steps

def save_db(cc_list, steps_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "config_codes.json"), "w") as f:
        json.dump(cc_list, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "steps.json"), "w") as f:
        json.dump(steps_list, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 配置码: {len(cc_list)} 条")
    print(f"✅ 操作步骤: {len(steps_list)} 条")
    print(f"✅ 保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    cc, steps = parse_docs()
    save_db(cc, steps)
