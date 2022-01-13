![example workflow](https://github.com/statneutrino/uscensus-fastapi/actions/workflows/python-app.yml/badge.svg)

This github repository contains an online API for a simple classification model
on the Census Income Data Set to predict salary. The API is live and deployed on Heroku and can be found at:
[https://uscensus-fastapi.herokuapp.com/](https://uscensus-fastapi.herokuapp.com/docs). This app is 
fast, type-checked and autodocumented API using FastAPI.

The machine learning is a very simple random forest classifier, and can be replaced easily with better models. However the point of this project was to:
- implement production frameworks such as Continuous Integration and Continuous Deployment
- ensure pipeliness pass unit tests before deployment
- testing of local and live API
- use of DVC (data version control with git).

This is a project completed as part of the Udacity Machine Learning
DevOps Engineer Nanodegree. 

### Use of remote storage

Models and data are stored in an AWS S3 bucket and pulled by DVC on Heroku when the API starts.

### CI/CD

Continuous Integration and Continuous Deployment (CI/CD) practices were used. Every commit push triggers a Github workflow, and unit tests using pytest are run before
master branch is automatically deployed to Heroku.

The badge above tracks whether CI is passing. More details can be found at the [Actions page](https://github.com/statneutrino/uscensus-fastapi/actions)

### Other files (for project rubric)

- Model performance on data slices for categories education, sex and race can be found at [slice_outputs/slice_output.txt](./slice_outputs/slice_output.txt)
- [Screenshot](./screenshots/example.png) of example json on FastAPI docs page

### Testing the live API using request module

### If you want to do the same - how to run on Heroku

We need to give Heroku the ability to pull in data from DVC upon app start up. We will install a [buildpack](https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt) that allows the installation of apt-files and then define the Aptfile that contains a path to DVC. I.e., in the CLI run:

`heroku buildpacks:add --index 1 heroku-community/apt`

Then in your root project folder create a file called `Aptfile` that specifies the release of DVC you want installed, e.g.
https://github.com/iterative/dvc/releases/download/2.0.18/dvc_2.0.18_amd64.deb
 
Add the following code block to your main.py:

```
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
```