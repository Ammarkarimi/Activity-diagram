# Web interface with Java and PlantUML, for Hugging Face Spaces or any Docker host.
#   docker build -t activity-diagram .
#   docker run -p 7860:7860 activity-diagram      ->  http://localhost:7860
FROM python:3.11-slim

# Java runs PlantUML; the fonts are needed to measure and draw diagram text.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless fontconfig fonts-dejavu-core libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN python -m scripts.install_plantuml

# Visitors enter their own OpenAI key; only the published runs are listed as saved runs.
ENV SAVED_RUNS_DIR="outputs/Agentic AI Results"

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
