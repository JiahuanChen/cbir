FROM binded/python-opencv
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt -e .
CMD ["python3", "-u", "bin/bow"]
