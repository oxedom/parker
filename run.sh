#!/bin/bash

# Set the directories to loop through
directories=(client server_flask server_node)

# Set the commands to run in each directory
commands=( "npm run start" "flask run" "nodemon app.js" )

# Loop through the directories and run the corresponding command
for i in ${!directories[@]}; do
  dir=${directories[$i]}
  command=${commands[$i]}
  cd "$dir"
  gnome-terminal -e "$command"
  cd ..
done