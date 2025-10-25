FROM python:3.10-slim

RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources


RUN apt-get update && apt-get install -y --no-install-recommends gcc procps vim ffmpeg



# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


WORKDIR /opt/rPPG

COPY ./requirements.txt /opt/rPPG/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /opt/rPPG/requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

CMD ["/bin/bash"]
