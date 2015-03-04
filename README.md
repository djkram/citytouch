# Citytouch Demo
Citytouch Project with Dockerfile ready to deploy!

##Steps:

1. Build Docker image: `docker build -t djkram/citytouch`
2. Deploy on Beanstalk: http://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create_deploy_docker.html

##Requirentments:

1. Exiting Postgis Database
2. Configure credentials en ENV variables: `[BD_NAME_CITYTOUCH,BD_HOST_CITYTOUCH,BD_USER_CITYTOUCH,BD_PASSWORD_CITYTOUCH]`

