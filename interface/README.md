# ELEC4907-Capstone
These files are used for running the project's interface.

It can either be run using python or by bulding the docker container
```
cd app;
python ./app.py

or

docker build -t capstone_interface .;
docker run -it -d -p 7860:7860 --name ELEC4907_Capstone_Interface capstone_interface:latest
```

To change the port either modify the ./app/app.py file or change the port via docker.