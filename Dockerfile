FROM python:3.7

WORKDIR /opt/omr

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python", "omr.py", "--help"]
