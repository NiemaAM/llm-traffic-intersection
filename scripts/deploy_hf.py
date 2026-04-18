"""
deploy_hf.py
------------
Deploy the Streamlit app to a Hugging Face Space.
Called from CI/CD pipeline on merge to main.
"""

import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE = os.environ["HF_SPACE"]  # e.g. "your-username/traffic-intersection"

api = HfApi(token=HF_TOKEN)

# Upload the Streamlit app as the main app.py
api.upload_file(
    path_or_fileobj="src/api/streamlit_app.py",
    path_in_repo="app.py",
    repo_id=HF_SPACE,
    repo_type="space",
    commit_message="Deploy updated Streamlit app",
)

# Upload requirements
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=HF_SPACE,
    repo_type="space",
    commit_message="Update requirements",
)

print(f"✅ Deployed to https://huggingface.co/spaces/{HF_SPACE}")
