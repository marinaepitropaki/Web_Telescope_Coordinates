FROM python:3.8.7-buster
#Make  a directory for our application
# WORKDIR /app
#Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

#Copy source code
# COPY . /app

#run the application
CMD ["python", "/app/web_telegraph.py"]