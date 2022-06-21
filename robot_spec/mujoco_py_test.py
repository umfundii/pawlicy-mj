#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from re import T
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import math

model = load_model_from_path("/home/karma/Documents/DL_Lab/Project/robot_spec/xml/a1_mj210.xml")
sim = MjSim(model)


viewer = MjViewer(sim)

sim_state = sim.get_state()


info_counter = 0
while True:
    if not info_counter:
        print(type(sim_state))


    sim.step()
    viewer.render()
    #print(" ")
    
    if os.getenv('TESTING') is not None:
        break
