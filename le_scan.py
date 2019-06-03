#! /usr/bin/env python
# -*- coding:utf-8 -*-


import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
import math

v = 0.1 #velocidade linear
w = math.pi/4 #velocidade angular
distance = []

def scaneou(dado):
	global distance
	distance = dado.ranges


	

if __name__== "__main__":

	rospy.init_node("le_scan")

	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

	velocidade = Twist(Vector3(0,0,0), Vector3(0,0,0))

	while not rospy.is_shutdown():

		
		for i in range(len(distance)):


			if distance[i] > 0 and distance[i] <= 0.2:

				if i <= 90:

					velocidade = Twist(Vector3(-v, 0, 0), Vector3(0, 0, w))

				elif i <= 180:

					velocidade = Twist(Vector3(v, 0, 0), Vector3(0, 0, w))

				elif i <= 270:

					velocidade = Twist(Vector3(v, 0, 0), Vector3(0, 0,w))

				elif i <= 360:

					velocidade = Twist(Vector3(-v, 0, 0), Vector3(0, 0, -w))

		# print(velocidade)

		velocidade_saida.publish(velocidade)
		velocidade = Twist(Vector3(0,0,0), Vector3(0,0,0))

		rospy.sleep(0.1)