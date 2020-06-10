### Collect raw data
First you should collect the images in CARLA. Here we manually controled the agent to collect the images.
Before running the code, a CARLA server should be started. Then you can run 
```
python utils/manual_control_collectdata0.8.2.py
```
It would automatically record the images with corresponding position in the map in './dataset/carla_rawdata/' <br>


### Train the networks
To train the CNN encoder and connect-predict-network, run 
```
python Encoder/cnn.py
```
The configurations are in 'config.py' and you may carefully check the settings before running.
If it is the first time to run, the dataset would be automatically made from the raw data. <br>


### Construct the topological map
After the networks are trained, the topological map can be constructed. 
To construct the topological map, you should specify the path of images that were collected when exploring the map by modifying '--path_images'.
Run 
```
python Encoder/ConstrucGraph.py
```
Then the topological map would be saved and it would also be visualized. <br>


### Train the DRL controller
The CARLA server should be started af first. Then you can run 
```
python train.py
```

