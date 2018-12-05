import sys
sys.path.append('C:/Users/Ashutosh/Documents/GitHub/EmoPy/src/')
from fermodel import FERModel

target_emotions = ['anger','happiness','calm']
model = FERModel(target_emotions,verbose=True)

print('Predicting on Image....')
model.predict('test/YM.HA2.53.jpg')