#!/bin/bash
# source .env

while true; do
  read -p "Do you want build NEXTJS? [y/n]: " yn
  case $yn in
    [Yy]* ) willBuildNext=true; break;;
    [Nn]* ) willBuildNext=false; break;;
    * ) echo "Please answer 'yes' or 'no'.";;
  esac
done


while true; do
  read -p "Do you want to build DOCKER? [y/n]: " yn
  case $yn in
    [Yy]* ) willBuildDocker=true; break;;
    [Nn]* ) willBuildDocker=false; break;;
    * ) echo "Please answer 'yes' or 'no'.";;
  esac
done

# Use the answer variable in your script
if [ "$willBuildNext" = true ]; then
  echo "Building NEXTJS project..."
  npm run build
  echo "Finished Building NextJS project"
fi

if [ "$willBuildDocker" = true ]; then
  echo "Building Docker Image..."
  sudo docker build -t oxedom/parker .


  echo "Finished Building Docker"
fi


echo "Pushing to Dockerhub"


sudo docker push oxedom/parker 



EOF