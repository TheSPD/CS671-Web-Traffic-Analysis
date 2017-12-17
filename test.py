from dataGenerator import DataGenerator

X = []
y = []

training_generator = DataGenerator().generate(range(543 * 50))
X, y = training_generator

print X, y
