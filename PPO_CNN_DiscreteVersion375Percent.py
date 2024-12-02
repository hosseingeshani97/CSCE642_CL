#this is a PPO algorithm, with cnn policy, continuous action space for steering
import glob
import os
import sys
import carla
import random
import time
import numpy as np
import cv2
import math
import argparse
import gymnasium as gym
from typing import Optional
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
#from gymnasium.wrappers import EnvCompatibility
import torch
tickRate = 0.25

# Ensure GPU usage
print("CUDA Available:", torch.cuda.is_available())

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class CarEnv:
    frr =1
    SHOW_CAM = None
    inpSize = 9
    STEER_AMT = 0.5
    im_width = None
    im_height = None
    front_camera= None
    SHOW_LIDAR = False
    frontLidar = None
    writeOnce = False
    Lidar_Depth = "25"
    Lidar_Resolution = 50
    Lidar_PPS = '32000'
    Lidar_RPS = '200'
    Lidar_Channels = '16'
    Points_Per_Observation = 10000
    loadingNewWorld = False
    DEFAULTTHROTTLE = None
    semLidarCamera = np.ones((100,100,1))
    
    

    def __init__(self,SHOW_PREVIEW = False, IM_WIDTH = 100,IM_HEIGHT = 100,DEFAULTTHROTTLE=0.4, LOADINGNEWWORLD = True):
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(20)
        self.loadingNewWorld = LOADINGNEWWORLD
        self.SHOW_LIDAR = SHOW_PREVIEW
        if  (self.loadingNewWorld):
            self.client.load_world("Town04")
            self.world = self.client.get_world()
            itemsToRemove = [carla.CityObjectLabel.TrafficLight,carla.CityObjectLabel.Fences,carla.CityObjectLabel.TrafficSigns,carla.CityObjectLabel.Vegetation,carla.CityObjectLabel.Poles,carla.CityObjectLabel.Buildings,carla.CityObjectLabel.Other]

            for item in itemsToRemove:
                env_objs = self.world.get_environment_objects(item)
                objectsToToggle = []
                for x in env_objs:
                    objectsToToggle.append(x.id)
                self.world.enable_environment_objects(objectsToToggle,False)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        self.world.apply_settings(settings)
        self.client.set_timeout(10)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = tickRate
        self.world.apply_settings(settings)
        self.DEFAULTTHROTTLE = DEFAULTTHROTTLE
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.Lidar_Field = int(self.Lidar_Depth) * 2 * self.Lidar_Resolution + 1
        

    def reset(self):
        #spawning
        self.destroyList = []
        approvedIndexes =[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        spawnIndex = random.randint(0,len(approvedIndexes)-1)
        spawnPointActual = approvedIndexes[spawnIndex]
        self.transform = (self.world.get_map().get_spawn_points())[spawnPointActual]
        spectator = self.world.get_spectator()
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.destroyList.append(self.vehicle)
        self.transform.location.z += 6
        self.transform.rotation.pitch -= 25
        spectator.set_transform(self.transform)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))

        #colSensors
        self.collision_hist = []
        colsensor = self.blueprint_library.find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x = 0, z=0.76))
        self.colsensor = self.world.spawn_actor(colsensor,transform,attach_to=self.vehicle)
        self.destroyList.append(self.colsensor)
        self.colsensor.listen(lambda event:self.collision_data(event))

        #lidarSensor
        lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("range",self.Lidar_Depth)
        lidar_bp.set_attribute("lower_fov", "0")  
        semLidarKeys = {'channels' : self.Lidar_Channels, 'range' : self.Lidar_Depth,  'points_per_second': self.Lidar_PPS, 'rotation_frequency': self.Lidar_RPS}
        for key in semLidarKeys:
            lidar_bp.set_attribute(key,semLidarKeys[key])
        transform = carla.Transform(carla.Location(x = 0, z=0.86))
        self.semLidar = self.world.spawn_actor(lidar_bp,transform,attach_to=self.vehicle)
        self.destroyList.append(self.semLidar)
        self.semLidar.listen(lambda data:self.process_semantic_lidar(data))

        return 1
    def process_semantic_lidar(self,point_cloud):
        disp_size = [100,100]
        lidar_range = 2.0*float(self.Lidar_Depth)
        points = np.frombuffer(point_cloud.raw_data,dtype= np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        #if self.SHOW_LIDAR:
        #    cv2.imshow("",lidar_img)
        #    cv2.waitKey(1)
        self.semLidarCamera = lidar_img[:,:,1]
    
    def collision_data(self,event):
        self.collision_hist.append(event)
    
    def step(self,action):
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.DEFAULTTHROTTLE,steer = action))
        reward = 0
        done=False

        #env code reward stuff here
        self.world.tick()
        time.sleep(0.01)
        done = True

        if len(self.collision_hist) != 0:
            
            done = True
            reward = -15
        else:
            reward += .1
        stateObservation = 1
        return self.semLidarCamera, reward,done,None
    
    def envEnd(self):
        
        vehicles = self.world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        vehicles = self.world.get_actors().filter("*sensor*")
        for vehicle in vehicles:
            
            if vehicle.is_alive:
                vehicle.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
    
    def envCleanEpisode(self):
        for x in self.destroyList:
            x.destroy()
        self.destroyList = []
        vehicles = self.world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            vehicle.destroy()
        vehicles = self.world.get_actors().filter("*sensor*")
        for vehicle in vehicles:
            vehicle.destroy()
        




class CarEnvGYM(gym.Env):
    frr =1
    SHOW_CAM = None
    inpSize = 9
    STEER_AMT = 0.5
    stepVariable = 0
    im_width = None
    im_height = None
    front_camera= None
    SHOW_LIDAR = False
    frontLidar = None
    writeOnce = False
    Lidar_Depth = "28"
    Lidar_Resolution = 50
    Lidar_PPS = '1000000'
    destroyList = []
    Lidar_RPS = '8'
    Lidar_Channels = '128'
    Points_Per_Observation = 10000
    loadingNewWorld = False
    DEFAULTTHROTTLE = 0.4
    semLidarCamera = np.ones((150,150,3))
    epRewards = []
    currentRewards = 0
    totalTimeSteps = 0
    cl1YGoal = -234
    clY2Goal = -234
    clY3Goal = -166
    clY4Goal = -116
    clY5Goal = -42

    cl1Graph = []
    cl2Graph = []
    cl3Graph = []
    cl4Graph = []
    cl5Graph = []
    cl6Graph = []
    
    mySpawnPoint = 0
    
    curriculumLearning = False

    def __init__(self,totalTimeSteps):
        self.totalTimeSteps = totalTimeSteps
        
        super(CarEnvGYM,self).__init__()
        self.action_space = gym.spaces.Discrete(3,start=-1)
        #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(150, 150, 3), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(150, 150, 3), dtype=np.uint8)
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(20)
        if  (self.loadingNewWorld):
            self.client.load_world("Town04")
            self.world = self.client.get_world()
            itemsToRemove = [carla.CityObjectLabel.TrafficLight,carla.CityObjectLabel.Fences,carla.CityObjectLabel.TrafficSigns,carla.CityObjectLabel.Vegetation,carla.CityObjectLabel.Poles,carla.CityObjectLabel.Buildings,carla.CityObjectLabel.Other]

            for item in itemsToRemove:
                env_objs = self.world.get_environment_objects(item)
                objectsToToggle = []
                for x in env_objs:
                    objectsToToggle.append(x.id)
                self.world.enable_environment_objects(objectsToToggle,False)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        self.world.apply_settings(settings)
        self.maxSteps = 0
        self.client.set_timeout(10)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = tickRate 
        self.world.apply_settings(settings)
        
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.Lidar_Field = int(self.Lidar_Depth) * 2 * self.Lidar_Resolution + 1

        blueprint_library = self.world.get_blueprint_library()

        # Look for barrel assets (search for anything related to 'barrel')
        

        allIds = []
        vehicles = self.world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        for blueprint in blueprint_library.filter('static.prop.trafficwarning'):
            allIds.append(blueprint.id)
        try:
            for i in range(len(allIds)):
                print(i,allIds[i])
                self.barrel_bp = blueprint_library.filter('static.prop.trafficwarning')[i]
                self.transform = (self.world.get_map().get_spawn_points())[i]
                self.transform.location.x += 35
                self.transform.location.y += 3
                self.transform.location.z -=0.25
                self.world.spawn_actor(self.barrel_bp,self.transform)
                print("------")
        except Exception as e:
            print(e)
        spectator = self.world.get_spectator()
        actualBarrelBP = self.blueprint_library.filter('static.prop.trafficc*')[1]
        listXBarrelSpawns = [378.43,395.48,376.8]
        listYBarrelSpawns = [-182.09,-132.38,-76.31]
        for i in range(len(listXBarrelSpawns)):
            for j in range(15):
                self.transform.location.x = listXBarrelSpawns[i] 
                self.transform.location.y = listYBarrelSpawns[i] + 2*j
                
                self.transform.rotation.yaw = 90
                #spectator.set_transform(self.transform)
                self.world.spawn_actor(self.barrel_bp,self.transform)
                self.transform.location.z += 0.9
                self.transform.location.x -= 0.5
                self.world.spawn_actor(actualBarrelBP,self.transform)
                self.transform.location.x += 1
                self.world.spawn_actor(actualBarrelBP,self.transform)
                self.transform.location.x -=0.5 
                self.world.spawn_actor(actualBarrelBP,self.transform)
                self.transform.location.z -=0.9


        listXBarriers = [394.4,380]
        listYBarriers = [-166,-116]
        actualBarrelBP = self.blueprint_library.filter('static.prop.trafficc*')[1]
        for i in range(len(listYBarriers)):
            for j in range(2):
                self.transform.location.y = listYBarriers[i] 
                self.transform.location.x = listXBarriers[i] - 2*j
                self.transform.location.z = 0.05
                self.transform.rotation.yaw = 10
                #spectator.set_transform(self.transform)
                self.world.spawn_actor(self.barrel_bp,self.transform)
                self.transform.location.z += 0.9
                self.transform.location.x -= 0.5
                self.world.spawn_actor(actualBarrelBP,self.transform)
                self.transform.location.x += 1
                self.world.spawn_actor(actualBarrelBP,self.transform)
                self.transform.location.x -=0.5 
                self.world.spawn_actor(actualBarrelBP,self.transform)
                

        #spawn enemy vehicle
                







        listXRandomBarrels = [296,342,389,391.8,395.07,347.3]
        listYRandomBarrels = [-361,-327,-223,-68,-68.8,14.91]

        for i in range(len(listXRandomBarrels)):
            break
            self.transform.location.x = listXRandomBarrels[i] 
            self.transform.location.y = listYRandomBarrels[i]
            spectator.set_transform(self.transform)
            self.world.spawn_actor(self.barrel_bp,self.transform)
       
        

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        #spawning
        if seed is not None:
            np.random.seed(seed)
        #if len(self.destroyList) > 0:
        #    for xy in self.destroyList:
        #        xy.destroy()
        #        print("just destroyed a vehicle")
        self.destroyList = []
        #self.epRewards.append(self.currentRewards)
        #self.currentRewards = 0
        vehicles = self.world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            break
            if vehicle.is_alive:
                vehicle.destroy()
    
        self.mySpawnPoint = 1
        for i in range(1,7):
            if self.stepVariable < ((self.totalTimeSteps / 6) * i):
                
                break
            else:
                self.mySpawnPoint += 1
        if self.mySpawnPoint >6:
            self.mySpawnPoint = 6
        self.transform = (self.world.get_map().get_spawn_points())[0]
        spawnRotationVariance = 8

        
        lastThreshold = -5
        successValue = 0.6
        if not self.curriculumLearning:
            self.mySpawnPoint = 6 #this will cause the agent to spawn at the start
        if self.mySpawnPoint == 1:
            
            self.transform = (self.world.get_map().get_spawn_points())[0]
            self.transform.location.x = 385
            self.transform.location.y = -245
            self.transform.rotation.pitch = 0
            self.transform.rotation.yaw = 120.0 + np.random.randint(-1* spawnRotationVariance,spawnRotationVariance)

            
            if len(self.cl1Graph) > abs(lastThreshold):
                average = sum(self.cl1Graph[lastThreshold:]) /len(self.cl1Graph[lastThreshold:])
                if average >= successValue:
                    print(f"it achieved a good avg to skip past this CL{self.mySpawnPoint}")
                    self.mySpawnPoint += 1 

            
        if self.mySpawnPoint==2:
            self.transform = (self.world.get_map().get_spawn_points())[0]
            self.transform.location.x = 391
            self.transform.location.y = -245
            self.transform.rotation.pitch = 0
            self.transform.rotation.yaw = 65.0 + np.random.randint(-1* spawnRotationVariance,spawnRotationVariance)
            
            if len(self.cl2Graph) > abs(lastThreshold):
                average = sum(self.cl2Graph[lastThreshold:]) /len(self.cl2Graph[lastThreshold:])
                if average >= successValue:
                    print(f"it achieved a good avg to skip past this CL{self.mySpawnPoint}")
                    self.mySpawnPoint += 1 

        if self.mySpawnPoint == 3:
            self.transform = (self.world.get_map().get_spawn_points())[0]
            self.transform.location.x = 393.1
            self.transform.location.y = -188
            self.transform.rotation.pitch = 0
            self.transform.rotation.yaw = 90.0

            
            if len(self.cl3Graph) > abs(lastThreshold):
                average = sum(self.cl3Graph[lastThreshold:]) /len(self.cl3Graph[lastThreshold:])
                if average >= successValue:
                    print(f"it achieved a good avg to skip past this CL{self.mySpawnPoint}")
                    self.mySpawnPoint += 1 
            
        if self.mySpawnPoint == 4:
            self.transform = (self.world.get_map().get_spawn_points())[0]
            self.transform.location.x = 385
            self.transform.location.y = -153
            self.transform.rotation.pitch = 0
            self.transform.rotation.yaw = 90.0

            
            if len(self.cl4Graph) > abs(lastThreshold):
                average = sum(self.cl4Graph[lastThreshold:]) /len(self.cl4Graph[lastThreshold:])
                if average >= successValue:
                    print(f"it achieved a good avg to skip past this CL{self.mySpawnPoint}")
                    self.mySpawnPoint += 1 

        if self.mySpawnPoint == 5:
            self.transform = (self.world.get_map().get_spawn_points())[0]
            self.transform.location.x = 387
            self.transform.location.y = -90
            self.transform.rotation.pitch = 0
            self.transform.rotation.yaw = 90.0

            
            if len(self.cl5Graph) > abs(lastThreshold):
                average = sum(self.cl5Graph[lastThreshold:]) /len(self.cl5Graph[lastThreshold:])
                if average >= successValue:
                    print(f"it achieved a good avg to skip past this CL{self.mySpawnPoint}")
                    self.mySpawnPoint += 1 
        if not self.curriculumLearning:
            self.mySpawnPoint = 6 
        if self.mySpawnPoint == 6:
            self.transform.location.x = 383
            self.transform.location.y = -245
            self.transform.rotation.pitch = 4
            self.transform.rotation.yaw = 91.3

        
            
    
        #print(f"Spawn point value in reset: {self.mySpawnPoint}")

        #print(self.transform)
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.transform = (self.world.get_map().get_spawn_points())[0]
        self.transform.location.x = 389
        self.transform.location.y = -28
        self.transform.rotation.yaw = -90
        self.autopilot = self.world.spawn_actor(self.model_3,self.transform)
        self.autopilot.apply_control(carla.VehicleControl(throttle = 0.2))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.8,steer = 0))
        




        self.transform.location.x = 380
        self.transform.location.y = -281
        #self.destroyList.append(self.vehicle)
        self.transform.location.z += 6
        self.transform.rotation.pitch -= 25
        #spectator.set_transform(self.transform)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))

        #colSensors
        self.collision_hist = []
        colsensor = self.blueprint_library.find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x = 0, z=0.99))
        self.colsensor = self.world.spawn_actor(colsensor,transform,attach_to=self.vehicle)
        self.destroyList.append(self.colsensor)
        self.colsensor.listen(lambda event:self.collision_data(event))

        #lidarSensor
        lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("range",self.Lidar_Depth)
        lidar_bp.set_attribute("lower_fov", "-80") 
        lidar_bp.set_attribute("upper_fov", "1") 
        semLidarKeys = {'channels' : self.Lidar_Channels, 'range' : self.Lidar_Depth,  'points_per_second': self.Lidar_PPS, 'rotation_frequency': self.Lidar_RPS}
        for key in semLidarKeys:
            lidar_bp.set_attribute(key,semLidarKeys[key])
        self.lidarZValue = 1.86
        transform = carla.Transform(carla.Location(x = 0, z=self.lidarZValue))
        self.semLidar = self.world.spawn_actor(lidar_bp,transform,attach_to=self.vehicle)
        self.destroyList.append(self.semLidar)
        self.semLidar.listen(lambda data:self.process_semantic_lidar(data))
        observation = np.random.randint(0, 255, size=(150, 150,3))
        info = {}
        
        return observation, info
    def process_semantic_lidar(self,point_cloud):
        disp_size = [150,150]
        lidar_range = 2.0*float(self.Lidar_Depth)
        points = np.frombuffer(point_cloud.raw_data,dtype= np.dtype('f4'))

        points = np.reshape(points, (int(points.shape[0] / 6), 6))

        points = np.array(points[:, :3])
        lidar_data = points[points[:, 2] >= -self.lidarZValue + 0.1, :2]
        #lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))

        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.SHOW_LIDAR or True:
            pass
            #cv2.imshow("",lidar_img)
            #cv2.waitKey(1)
        #self.semLidarCamera = lidar_img[:,:,1] #for mlp
        
        self.semLidarCamera = lidar_img # for cnn
        
    
    def collision_data(self,event):
        self.collision_hist.append(event)
    
    def step(self,action):
        self.stepVariable += 1
        #print(action,type(action))
        #print(action[0],type(action[0]))
        if action == 0:
            action = -0.4
        elif action == 1:
            action = 0
        elif action ==2:
            action = 0.4
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.4,steer =action))
        reward = 0
        done=False

        #env code reward stuff here
        self.world.tick()
        targetX = 385
        targetY = -19.9

        vehicleLocation = self.vehicle.get_location()
        distanceToTarget = math.sqrt((targetX - vehicleLocation.x)**2 + (targetY - vehicleLocation.y)**2)
        reward = 0


        if len(self.collision_hist) != 0:
            done = True
            reward = -1
        if distanceToTarget < 10:
            done = True
            print("SUCCESS!")
            reward = 1
        if self.mySpawnPoint == 1:
            if self.vehicle.get_location().y > self.cl1YGoal:
                done = True
                self.cl1YGoal += 0.25
                print(f"succes for CL {self.mySpawnPoint}")
                reward = 1
            if done:
                self.cl1Graph.append(reward)
        elif self.mySpawnPoint == 2:
            if self.vehicle.get_location().y > self.clY2Goal: #-234 start goal
                done = True
                self.clY2Goal += 0.25
                print(f"succes for CL {self.mySpawnPoint}")
                reward = 1
            if done:
                self.cl2Graph.append(reward)
        elif self.mySpawnPoint == 3:
            if self.vehicle.get_location().y > self.clY3Goal:
                done = True
                print(f"succes for CL {self.mySpawnPoint}")
                reward = 1
                self.clY3Goal += 0.25
            if done:
                self.cl3Graph.append(reward)
        elif self.mySpawnPoint == 4:
            if self.vehicle.get_location().y > self.clY4Goal:
                done = True
                print(f"succes for CL {self.mySpawnPoint}")
                reward = 1
                self.clY4Goal += 0.25
            if done:
                self.cl4Graph.append(reward)
        elif self.mySpawnPoint == 5:
            if self.vehicle.get_location().y > self.clY5Goal: #-42
                done = True
                self.clY5Goal += 0.25
                print(f"succes for CL {self.mySpawnPoint}")
                
                reward = 1
            if done:
                self.cl5Graph.append(reward)
        elif self.mySpawnPoint == 6:
            if self.vehicle.get_location().y > -40:
                done = True
                print(f"succes for CL {self.mySpawnPoint}")
                reward = 1
            if done:
                self.cl6Graph.append(reward)
        #print(f"Spawn point value in step: {self.mySpawnPoint}")
        aa = {}
        if done:
            self.autopilot.destroy()
            self.vehicle.destroy()
            self.colsensor.destroy()
            self.semLidar.destroy()
            self.currentRewards = distanceToTarget
        
        return self.semLidarCamera, reward,done,None,aa
    
    def envEnd(self):
        
        vehicles = self.world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        vehicles = self.world.get_actors().filter("*sensor*")
        for vehicle in vehicles:
            
            if vehicle.is_alive:
                vehicle.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
    
    def envCleanEpisode(self):
        for x in self.destroyList:
            x.destroy()
        self.destroyList = []
        vehicles = self.world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            vehicle.destroy()
        vehicles = self.world.get_actors().filter("*sensor*")
        for vehicle in vehicles:
            vehicle.destroy()
    def moving_average(self,values, window_size):
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')
    def makePlot(self):
        window_size = 5
        
        #print("ep rewards", self.epRewards)
        # Plot the data and the moving average
        allRewards = [self.cl1Graph,self.cl2Graph,self.cl3Graph,self.cl4Graph,self.cl5Graph,self.cl6Graph]

        for k in range(len(allRewards)):
            try:
                reward = allRewards[k]
                
                title = "Curriculum_Stage " + str(k + 1)
                if k == 5:
                    title = "Final_Stage Reward Plot"
                moving_avg = self.moving_average(reward, window_size)
                #title="Final_Stage Reward Plot, nonCL" #comment this out if you want curriculum learning. Un-comment this out if you do not want curriculum learning.
                plt.plot(reward, label="Original Data", marker="o")
                plt.plot(range(window_size - 1, len(reward)), moving_avg, label="Moving Average", color="orange", marker="x")
                plt.title(title)
                plt.xlabel("Episode")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.savefig(title + ".png")
                plt.show()
            except:
                pass


   

def main():
    try:
        parser = argparse.ArgumentParser(description="Process some flags and arguments.")
        # Adding optional arguments with flags
        parser.add_argument('-s', '--sleep', type=int, help="Your name", required=True)
        parser.add_argument('-a', '--age', type=int, help="Your age", required=False)
        parser.add_argument('-l','--loadNewWorld',type=bool, default=False, help="Your nwe world", required=False)
        parser.add_argument("-p",'--preview', type=bool, default=False)
        parser.add_argument("-n",'--numEpisodes', type=int, default=15)
        parser.add_argument("-e",'--epsilonInitial', type=float, default=0.9)
        parser.add_argument("-d",'--discount', type=float, default=0.99)
        parser.add_argument("-m",'--maxSteps', type=int, default=1000)
        parser.add_argument("-t",'--throttle', type=float, default=0.3)
        parser.add_argument("-q",'--timeSteps', type=int, default=25000)
        args = parser.parse_args()
        
        SHOW_PREVIEW = args.preview
        IM_HEIGHT = 100
        IM_WIDTH = 100
        REPLAY_MEMORY_SIZE = 5_000
        MIN_REPLAY_MEMORY_SIZE = 1_000
        MINIBATCH_SIZE = 16
        PREDICTION_BATCH_SIZE=1
        TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
        UPDATE_TARGET_EVERY = 5
        DISCOUNT =args.discount
        EPISODES = args.numEpisodes
        EPSILON = args.epsilonInitial
        EPSILON_DECAY = 0.95
        MIN_EPSILON = 0.001
        DEFAULTTHROTTLE = args.throttle
        AGGREGATE_STATS_EVERY = 10
        maxSteps = args.maxSteps
        if 2== 2:

            env = CarEnv(SHOW_PREVIEW=SHOW_PREVIEW,LOADINGNEWWORLD=args.loadNewWorld)
            
            time.sleep(3)
            try:
                for i in range(EPISODES):
                    env.reset()
                    totalReward = 0
                    for j in range(maxSteps):
                        action = 0.4
                        stateObservation, reward,done,_ = env.step(action=action)
                        totalReward += reward
                        
                        if done:
                            break
                    
                    env.envCleanEpisode()
                    
            except Exception as e:
                print(e)
    finally:
        y = 2

    try:
        myenv = CarEnvGYM(args.timeSteps)
        #obs = myenv.reset()
        #myenv = EnvCompatibility(myenv)

        model = PPO("CnnPolicy",myenv,verbose=1,device="cpu")
        model.learn(total_timesteps=args.timeSteps)
        model.save("fourthModelwCL")
        myenv.envCleanEpisode()
        myenv.envEnd()
        myenv.makePlot()
        
        

    finally:
        pass

    

if __name__ == "__main__":

    main()
    