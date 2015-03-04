#!/bin/bash

# starts sercices
echo "Starting Citytouch Demo !!!"

# Pico
service pico start

# Apache2
echo "Starting Apache2"
source /etc/apache2/envvars 
[ ! -d ${APACHE_RUN_DIR:-/var/run/apache2} ] && mkdir -p ${APACHE_RUN_DIR:-/var/run/apache2}
[ ! -d ${APACHE_LOCK_DIR:-/var/lock/apache2} ] && mkdir -p ${APACHE_LOCK_DIR:-/var/lock/apache2} && chown ${APACHE_RUN_USER:-www-data} ${APACHE_LOCK_DIR:-/var/lock/apache2}

echo "Citytouch running ;)"
exec /usr/sbin/apache2 -DFOREGROUND >> /var/log/apache.log 2>&1