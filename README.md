<div align="center">
	<h1> Neural Network Coursework </h1>
    This is some of my work that I did in my neural network course. I've got to learn a lot about perceptrons and complex CNN models. In this repository, you can find models that I've created for cifar10 (objects dataset) and face dataset for recognition. 
</div>

## How to run locally
Although I'll recommend to run these files on Google Colab follow these steps to run the files in your local. machine. 

### Clone this repository
```bash
git clone https://github.com/navkar98/nn-coursework.git
```

### Create a virtual environment and activate it
#### For windows
```bash
python -m venv env
env\Scripts\activate
```
#### For ubuntu
```bash
python -m venv env
source env\bin\activate
```

### Install all dependencies
```bash
pip install requirements.txt
```

### Run the file
```bash
python cifar10.py
python face_recognition.py
```