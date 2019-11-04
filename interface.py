from main import *
import speech_recognition as sr
r = sr.Recognizer()

print('Describe Your Model')
with sr.Microphone() as source:
	audio = r.listen(source)
try:
	# print('System predicts: ', r.recognize_google(audio))





	s = 'make me a segmentation network using the buildings dataset, relu nonlinearities, and 4 output classes, and input size of 192'
	string = r.recognize_google(audio)
	print(string)
	johnny(string)









except Exception:
	print('No audio detected')




# string = 'create a semantic segmentation network with 4 convolutional layers, an image input size of 64 pixels, and 4 output classes. train it on 100 epochs using the binary cross entropy loss function'

