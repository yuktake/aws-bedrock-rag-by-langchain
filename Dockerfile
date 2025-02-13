FROM python:3.11.3

# Pythonの出力表示をDocker用に設定
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python -
# Poetryのパスの設定
ENV PATH /root/.local/bin:$PATH

RUN poetry config virtualenvs.create false

COPY ./app/ .