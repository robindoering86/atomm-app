# atomm web app

This demo is a visualization tool for my capstone project "Enrich yourself" - Using Machine Learning to Optimize Stock Trading Strategies (https://github.com/robindoering86/capstone_nf) for neue fische Data Science Bootcamp.
It is a Dash web application used to visualize historical stock data from New York Stock Exchange and 20 technical chart indicators. In addition, it previously trained ML algorithms can be backtested on any stock in the database.   

## Running locally

To run a development instance locally, create a virtualenv, install the 
requirements from `requirements.txt` and launch `app.py` using the 
Python executable from the virtualenv.

## Deploying on ECS

Use `make image` to create a Docker image. Then, follow [these 
instructions](https://www.chrisvoncsefalvay.com/2019/08/28/deploying-dash-on-amazon-ecs/) 
to deploy the image on ECS.
