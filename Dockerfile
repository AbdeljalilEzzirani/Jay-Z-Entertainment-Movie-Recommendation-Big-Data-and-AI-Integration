FROM bitnami/spark:3.5.1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["tail", "-f", "/dev/null"]  # Keeps container running for debugging