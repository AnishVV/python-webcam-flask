This is a toy project for learning how to use a couple of python libraries. 

### What it does

1) The web client sends video stream data (from the user's webcam) to a flask server using socketio
2) The server does some processing on the video stream
3) The client receives the processed video stream and re-displays the results in a different frame

In the demo site, the server is simply flipping the image horizontally. You could imagine it doing something more sophisticated (e.g. applying some filters), but obviously I was too lazy to implement anything cool.

### Demo
[Live Demo](https://python-stream-video.herokuapp.com)

### Setup

- setup heroku (`brew tap heroku/brew && brew install heroku`)
- Use a python virtualenv
- `pip install -r requirements.txt`
- `heroku local`

- in your browser, navigate to localhost:5000

### Deploy to heroku

- `git push heroku master`
- heroku open

### Common Issues

If you run into a 'protocol not found' error, see if [this stackoverflow answer helps](https://stackoverflow.com/questions/40184788/protocol-not-found-socket-getprotobyname).
