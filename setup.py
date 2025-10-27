from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

exec(open("streamlit_langgraph/version.py").read())

setup(
    name="streamlit-langgraph",
    version=__version__,
    author="Jong Ha Shin",
    author_email="shinjh1206@gmail.com",
    description="A Streamlit package for building multiagent web interfaces with LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JjongX/streamlit-langgraph",
    packages=find_packages(exclude=["examples*"]),
    install_requires=[
        "streamlit>=1.50.0",
        "langchain>=1.0.1",
        "langgraph>=1.0.1",
        "langchain-openai>=1.0.0",
        "openai>=2.3.0",
        "typing-extensions>=4.15.0",
    ],
    extras_require={
        "viz": [
            "pygraphviz>=0.20.0",
            "streamlit-mermaid>=0.3.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="streamlit langgraph multiagent ai chatbot llm",
    license="MIT",
    include_package_data=True,
)
