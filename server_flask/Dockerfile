FROM python:3.7.2

WORKDIR /app

# Copy the current directory contents into the container at /app 
ADD . /app
RUN chmod +x app.py

ENV FLASK_APP=app.py
RUN apt-get update
RUN apt-get install 'libgl1-mesa-dev' -y
# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["uwsgi", "app.ini"]