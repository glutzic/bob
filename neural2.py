import pprint
import turtle
import time
import math
import random

turtle.shape("circle")
turtle.turtlesize(0.5, 0.5)
turtle.ht()
turtle.penup()
turtle.speed(0)
turtle.setup(1200, 600)
turtle.tracer(50, 0)

class Network:
	"""docstring for Network"""
	def __init__(self):
		self.layers = [Layer("input"), Layer("hidden"), Layer("output")]

	def getLayer(self, layerName):
		for layer in self.layers:
			if layer.name == layerName:
				return layer

		return

	def addNeuron(self, layerName, neuron):
		if not neuron in self.getLayer(layerName).neurons:
			self.getLayer(layerName).addNeuron(neuron)

	def getNeuronsFromLayer(self, layerName):
		return self.getLayer(layerName).neurons

	def getAllNeurons(self):
		allNeurons = []

		for layer in self.layers:
			allNeurons.extend(layer.neurons)

		return allNeurons

	def isNeuronInLayer(self, neuron, layerName):
		for networkNeuron in self.getNeuronsFromLayer(layerName):
			if networkNeuron == neuron:
				return True

		return False

	def getAllConnections(self):
		allConnections = []

		for neuron in self.getAllNeurons():
			for inputConnection in neuron.inputConnections:
				alreadyAdded = False
				
				for connection in allConnections:
					if connection.equals(inputConnection):
						alreadyAdded = True
						break

				if not alreadyAdded:
					allConnections.append(inputConnection)

			for outputConnection in neuron.outputConnections:
				alreadyAdded = False

				for connection in allConnections:
					if connection.equals(outputConnection):
						alreadyAdded = True
						break

				if not alreadyAdded:
					allConnections.append(outputConnection)

		return allConnections


class Layer:
	def __init__(self, name):
		self.name = name
		self.neurons = []
	def addNeuron(self, neuron):
		self.neurons.append(neuron)

class DrawableObject:
	def __init__(self, type, penColor = "black", fillColor = "black"):
		self.type = type
		self.penColor = penColor
		self.fillColor = fillColor

	def getType(self):
		return self.type;

	def draw(self, turtle):
		self.beforeDraw(turtle)
		self.doDraw(turtle)
		self.afterDraw(turtle)
		return

	def doDraw(self, turtle):
		return

	def beforeDraw(self, turtle):
		turtle.color(self.penColor, self.fillColor)
		return

	def afterDraw(self, turtle):
		return

class DrawableCircle(DrawableObject):
	def __init__(self, x = 0, y = 0, radius = 10, penColor = "black", fillColor = "black"):
		DrawableObject.__init__(self, 'circle', penColor, fillColor)
		self.x = x
		self.y = y
		self.radius = radius

	def doDraw(self, turtle):
		turtle.pu()
		turtle.setpos(self.x, self.y)
		turtle.begin_fill()
		turtle.circle(self.radius)
		turtle.end_fill()
		return

class DrawableLine(DrawableObject):
	def __init__(self, x1 = 0, y1 = 0, x2 = 0, y2 = 0, penColor = "black", fillColor = "black"):
		DrawableObject.__init__(self, 'line', penColor, fillColor)
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def doDraw(self, turtle):
		turtle.pu()
		turtle.setpos(self.x1, self.y1)
		turtle.pd()
		turtle.goto(self.x2, self.y2)
		turtle.pu()
		return

class Neuron(DrawableCircle):
	def __init__(self, value = 0.0):
		DrawableCircle.__init__(self, 0, 0, 10, "black", "gray")
		self.value = value
		self.inputConnections = []
		self.outputConnections = []
		self.biasValue = -1
		self.biasWeight = 0.5
		self.error = 0

	def addInputConnection(self, neuron):
		connection = Connection(neuron, self)
		
		if not self.hasInputConnection(connection):
			self.inputConnections.append(connection)
			neuron.outputConnections.append(connection)

	def addOutputConnection(self, neuron):
		connection = Connection(self, neuron)
		
		if not self.hasOutputConnection(connection):
			self.outputConnections.append(connection)
			neuron.inputConnections.append(connection)
	
	def hasInputConnection(self, connection):
		for inputConnection in self.inputConnections:
			if connection.equals(inputConnection):
				return True
		return False

	def hasOutputConnection(self, connection):
		for outputConnection in self.outputConnections:
			if connection.equals(outputConnection):
				return True
		return False

	def doDraw(self, turtle):
		DrawableCircle.doDraw(self, turtle)

		# Tutaj mozesz sobie dorysowywac wszystko zwiazane z neuronem
		turtle.write("V: " + str(self.value) + " | Er: " + str(self.error) + " | Bw: " + str(self.biasWeight) + " | Bv: " + str(self.biasValue))

	def calculateOutput(self):
		# Suma inputow
		inputTotal = 0

		if len(self.inputConnections):
			for inputConnection in self.inputConnections:
				inputTotal += inputConnection.input.calculateOutput() * inputConnection.weight
		else:
			return self.value

		# Odejmij bias
		inputTotal += self.biasValue * self.biasWeight
		
		# Activation function - sigmoid
		self.value = 1 / (1 + math.exp(inputTotal * -1))

		# Sigmoid rounding
		if self.value > 0.9999:
			self.value = 1

		if self.value < 0.0001:
			self.value = 0

		# Funkcja aktywacyjna
		return self.value

	def calculateError(self, desiredValue):
		error = 0

		if not len(self.outputConnections):
			error = self.value * (1 - self.value) * (desiredValue - self.value)
		else:
			outputErrorTotal = 0

			for outputConnection in self.outputConnections:
				outputErrorTotal += outputConnection.output.calculateError(desiredValue)

			error = self.value * (1 - self.value) * outputErrorTotal

		self.error = error

		return error

	def updateWeight(self, learningRate = 0.5):
		self.biasWeight += learningRate * self.biasValue * self.error



class Connection(DrawableLine):
	def __init__(self, input, output, weight = 0.5, x1 = 0, y1 = 0, x2 = 0, y2 = 0, penColor = "black", fillColor = "black"):
		DrawableLine.__init__(self, 0, 0, 0, 0, "black", "black")
		self.weight = weight
		self.input = input
		self.output = output

	def equals(self, connection):
		return self.input == connection.input and self.output == connection.output

	def doDraw(self, turtle):
		DrawableLine.doDraw(self, turtle)

		turtle.goto((self.x1 + self.x2)/2, (self.y1 + self.y2)/2)
		turtle.write("W: " + str(self.weight))
		turtle.goto((self.x1 + self.x2)/2, (self.y1 + self.y2)/2)
		turtle.pd()
		turtle.color("red")
		turtle.goto(self.x2, self.y2)
		turtle.pu()

	def updateWeight(self, learningRate = 0.5):
		self.weight += learningRate * self.input.value * self.output.error


class NetworkBuilder:
	def __init__(self):
		pass
	def build(self):
		pass #finish later :* rob
	def buildSimple(self, neuronsInInput, neuronsInHidden, neuronsInOutput):
		network = Network()

		for neurons in range(neuronsInInput):
			network.addNeuron('input', Neuron())

		for neurons in range(neuronsInHidden):
			network.addNeuron('hidden', Neuron())

		for neurons in range(neuronsInOutput):
			network.addNeuron('output', Neuron())

		for inputNeuron in network.getNeuronsFromLayer("input"):
			for hiddenNeuron in network.getNeuronsFromLayer("hidden"):
				inputNeuron.addOutputConnection(hiddenNeuron)

		for hiddenNeuron in network.getNeuronsFromLayer("hidden"):
			for outputNeuron in network.getNeuronsFromLayer("output"):
				hiddenNeuron.addOutputConnection(outputNeuron)

		return network

class Screen():
	def __init__(self, turtle, drawableObjects = []):
		self.drawableObjects = drawableObjects
		self.turtle = turtle.clone()

		# Turtle follower jest po to zeby podtrzymywac stary rysunek w momencie kiedy jest updatowany.
		# Nowy obrazek jest narysowany na stary, i stary usuwany
		self.turtleFollower = self.turtle.clone()
		self.height = self.turtle.getscreen().window_height()
		self.width = self.turtle.getscreen().window_width()

	def drawWith(self, turtle):
		self.height = turtle.getscreen().window_height()
		self.width = turtle.getscreen().window_width()

		for drawableObject in self.drawableObjects:
			drawableObject.draw(turtle)
		turtle.getscreen().update()

	def step(self, delay = 1):
		self.turtle.clear()
		self.drawWith(self.turtle)
		self.turtleFollower.clear()
		self.drawWith(self.turtleFollower)
		time.sleep(delay)

class UIHelper():
	def __init__(self, screen):
		self.screen = screen

	def tx(self, x):
		return x - (self.screen.width / 2)

	def ty(self, y):
		return -1 * (y - (self.screen.height / 2))

	def getGridPos(self, sliceX, sliceY, x, y):
		lenX = self.screen.width / sliceX
		lenY = self.screen.height / sliceY

		return {"x": (x * lenX) - (lenX / 2), "y": (y * lenY) - (lenY / 2)}

	def translateCoordinates(self, network):
		# Kalkulacja koordynatow kazdego obiektu na podstawie screena (jego wielkosci) - ustawianie x i y neuronow a pozniej polaczen

		# Loop po wszystkich neuronach i ustawienie ich lokacji
		layerCount = len(network.layers)

		for layerIndex, layer in enumerate(network.layers):
			for index, neuron in enumerate(layer.neurons):
				layerNeuronCount = len(layer.neurons)
				gridPos = self.getGridPos(layerCount, layerNeuronCount, layerIndex + 1, index + 1)
				#print("layer: " + str(layerIndex) + " neuron: " + str(index) + " gridpos: " + str(gridPos))
				neuron.x = self.tx(gridPos["x"])
				neuron.y = self.ty(gridPos["y"])

			layerIndex += 1

		# Loop po wszystkich polaczeniach i ustawienie ich coordynatow wzgledem ich input,output
		for connection in network.getAllConnections():
			connection.x1 = connection.input.x
			connection.y1 = connection.input.y + connection.input.radius
			connection.x2 = connection.output.x
			connection.y2 = connection.output.y + connection.output.radius
		return

class Trainer():
	def __init__(self):
		pass

	def train(self, network, inputs, desiredOutputs, learningRate = 0.5):
		# Set inputs
		for key, neuron in enumerate(network.getNeuronsFromLayer("input")):
			neuron.value = inputs[key]

		# Get network outputs
		outputs = []
		for key, neuron in enumerate(network.getNeuronsFromLayer("output")):
			outputs.append(neuron.calculateOutput())

		# Calculate neuron errors
		for key, neuron in enumerate(network.getNeuronsFromLayer("input")):
			neuron.calculateError(desiredOutputs[0])

		# Update weight for all connection
		for connection in network.getAllConnections():
			connection.updateWeight(learningRate)

		# Update bias weight for all neurons
		for neuron in network.getAllNeurons():
			neuron.updateWeight(learningRate)

	def run(self, network, inputs, desiredOutputs = []):
		# Set inputs
		for key, neuron in enumerate(network.getNeuronsFromLayer("input")):
			neuron.value = inputs[key]

		# Get network outputs
		outputs = []
		for key, neuron in enumerate(network.getNeuronsFromLayer("output")):
			outputs.append(neuron.calculateOutput())

		print(str(inputs) + " : " + str(outputs))


networkBuilder = NetworkBuilder()
network = networkBuilder.buildSimple(2,10,1)
screen = Screen(turtle, [])
trainer = Trainer()
i = 0

# Animacja lub wyswietlanie krokowe, moze byc nieskonczona
while 1 == 1:
	i += 1
	# Tutaj wszystkie zmiany w danym kroku. Np jakas ingerencja w strukture w czasie. Mozna dodawac obiekty do sieci i zmieniac ja.
	x = random.random()
	y = random.random()
	trainer.train(network, [x, y], [x + y], 0.1)

	if (i % 1000) == 0:
		print("Iteration: " + str(i))
		trainer.run(network, [0, 0])

	if (i % 100000) == 0:
		for x in range(1, 3):
			in1 = float(input("Enter input 1: "))
			in2 = float(input("Enter input 2: "))
			trainer.run(network, [in1, in2])

	# Pobieranie obiektow z sieci
	objects = []
	objects.extend(network.getAllNeurons())
	objects.extend(network.getAllConnections())

	# Tworzysz screen poprzez podanie mu wszystkich elementow ktore dziedzicza DrawableObject
	screen.drawableObjects = objects

	# Translacja koordynatow
	#UIHelper(screen).translateCoordinates(network)

	# Nastepny krok animacji, delay w sekundach
	#screen.step(0)

turtle.done()