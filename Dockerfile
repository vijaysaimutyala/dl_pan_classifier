FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD main.py main.py
ADD image_reader.py image_reader.py
ADD dl_or_pan.pth dl_or_pan.pth

# Run it once to trigger resnet download
RUN python main.py

EXPOSE 5000

# Start the server
CMD ["python", "main.py", "serve"]
