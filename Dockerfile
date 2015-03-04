FROM ubuntu:trusty
MAINTAINER Marc Planagumà <mplanaguma@bdigital.org>

# Init and update ubuntu
RUN apt-get -y update
RUN apt-get -y upgrade
RUN locale-gen --no-purge en_US.UTF-8
ENV LC_ALL en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

# Install Essentials
RUN apt-get install -y build-essential wget sysv-rc
RUN apt-get -y install supervisor

# Install Python 2.7
RUN apt-get install -y python libpq-dev python-dev python-setuptools
RUN apt-get install -y python-pip python-virtualenv
RUN apt-get install -y python-numpy python-scipy ipython cython
RUN apt-get install -y binutils libproj-dev gdal-bin

# Init Srcipts
VOLUME ["/app"]
ADD setup-env /app/setup-env

# Setup Pico
RUN pip install psycopg2 django greenlet gensim scikit-learn pico
RUN cd /tmp && wget --quiet --no-check-certificate https://github.com/surfly/gevent/archive/1.0rc2.tar.gz &&\
	tar -xf 1.0rc2.tar.gz && cd gevent-1.0rc2 &&\
	python setup.py build && python setup.py install
WORKDIR /app
COPY setup-env/pico /etc/init.d/pico
RUN chmod +x /etc/init.d/pico
RUN chmod +x /app/setup-env/pico_server
RUN update-rc.d pico defaults

# install Apache2
RUN apt-get install -y apache2

# add web
ADD web /var/www/html

# Configure Apache2
RUN echo "ProxyPass /pico/ http://localhost:8800/pico/ retry=5" > /etc/apache2/conf-available/proxy.conf
RUN echo "ProxyPassReverse /pico/ http://localhost:8800/pico/" >> /etc/apache2/conf-available/proxy.conf
RUN a2enconf proxy.conf
RUN a2enmod proxy_http

EXPOSE 80

RUN chmod +x /app/setup-env/start-citytouch.sh
ENTRYPOINT ["/app/setup-env/start-citytouch.sh"]
