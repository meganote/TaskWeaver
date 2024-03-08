FROM python:3.10

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY . /TaskWeaver
RUN cd TaskWeaver && \
    pip install -r requirements.txt && \
    pip install chainlit

WORKDIR /TaskWeaver/playground/UI

EXPOSE 8000

CMD ["chainlit", "run", "app.py"]
