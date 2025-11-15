from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kodikon",
    version="0.1.0",
    author="Kodikon Team",
    description="Distributed baggage-tracking system for 24-hour hackathon using P2P mesh network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "opencv-python>=4.8.1.78",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "ultralytics>=8.0.192",
        "onnxruntime>=1.16.3",
        "aiohttp>=3.8.5",
        "websockets>=11.0.3",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "pyyaml>=6.0.1",
        "pillow>=10.0.1",
    ],
)
