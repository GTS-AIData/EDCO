# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# generate_temp_config.py
import os
import sys
import re
from pathlib import Path


def load_and_replace(template_path: str) -> str:
    if not Path(template_path).exists():
        raise FileNotFoundError(f"Template config not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    def replace_env_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))

    content = re.sub(r'\$\{([A-Za-z0-9_]+)\}', replace_env_var, content)
    return content


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_temp_config.py <input_yaml> <output_yaml>")
        sys.exit(1)

    input_yaml = sys.argv[1]
    output_yaml = sys.argv[2]

    try:
        expanded_content = load_and_replace(input_yaml)
        Path(output_yaml).parent.mkdir(parents=True, exist_ok=True)
        with open(output_yaml, 'w', encoding='utf-8') as f:
            f.write(expanded_content)
        print(f"✅ Generated config: {output_yaml}")
    except Exception as e:
        print(f"❌ Failed to generate config: {e}")
        sys.exit(1)
