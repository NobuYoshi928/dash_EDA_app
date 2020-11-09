FROM python:3.7.3
USER root

RUN apt-get update
RUN apt-get install -y vim less
RUN apt-get install -y zsh less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install \
    dash==1.16.3 \
    pandas \
    category_encoders \
    scikit-learn

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

EXPOSE 5050
CMD ["bash"]
