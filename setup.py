from setuptools import setup, find_packages

setup(
    name="llm-traffic-intersection",
    version="1.0.0",
    description="LLM-Driven Agents for Traffic Intersection Conflict Resolution",
    author="CSC5382 – AI for Digital Transformation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "openai>=1.30.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.0",
        "mlflow>=2.12.0",
        "zenml>=0.60.0",
        "streamlit>=1.35.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "generate-data=src.data.generate_data:main",
            "run-api=src.api.app:main",
        ],
    },
)
