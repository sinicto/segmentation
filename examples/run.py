from segmentation import Unet, Loader

print('='*4, 'START')
loader = Loader()
train, dev, test = loader.load_split_data()
model = Unet()
model.fit(train)
model.store()

print('Train')
model.evaluate(train)

print('Dev') 
model.evaluate(dev)

print('Test')
model.evaluate(test)