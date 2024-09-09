#Выкачиваем из dockerhub образ с python версии 3.9
FROM python:3.11
#Устанавливаем рабочую директорию для проекта в контейнере
WORKDIR /backend
#Скачиваем/обновляем необходимые библиотеки для проекта
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY requirements.txt /backend
RUN pip3 install --upgrade pip -r requirements.txt --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org
COPY . /backend
#Устанавливаем порт, который будет использоваться для сервера
EXPOSE 8080
ENTRYPOINT ["streamlit"]