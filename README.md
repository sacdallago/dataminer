# Dataminer

1. TensorFlow

[Tensorflow](https://github.com/tensorflow/tensorflow) is a DeepLearning device (CNN - Convonutional Neural Network) developed by Google. It is platform-dependent, 
thus I suggest to run it using [docker](https://www.docker.com/).
With Docker it is fairly easy to get tensorflow up and running, you just need to:
	1. Install Docker on your system
	2. Clone this repository
	3. `cd` from your terminal in this repository
	4. Run a docker instance with the following
	```bash
	docker run -it -p 3030:8888 -p 6006:6006 -v /$(pwd):/notebooks gcr.io/tensorflow/tensorflow
	```
	If you are running windows, change `/$(pwd)` to `$(pwd)`

After this, you will see an interactive session on your terminal that will tell you that `jupyter` is running. You can connect from your browser to `localhost:3030` 
to see the jupyter instance.

**Important**: To run the `jupyter` notebook again the next time, you will only need to execute the command `docker start -i tf` from the command line.
