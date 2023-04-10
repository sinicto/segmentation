from segmentation import Unet, Loader

print('='*4, 'START')
default_data_path = 'drive/MyDrive/data/CVPPPSegmData/'
default_split_path = 'drive/MyDrive/data/CVPPPSegmData/split.csv'
loader = Loader(split_path=default_split_path,
                 data_path=default_data_path)
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