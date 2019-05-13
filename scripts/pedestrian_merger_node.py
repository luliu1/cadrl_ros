#!/usr/bin/env python

import rospy
import sys
from obstacle_detector.msg import Obstacles
from navigation_msgs.msg import Pedestrians
from people_msgs.msg import Person

class PedestrianPublisher():
    def __init__(self):
        self.node_name = rospy.get_name()
        self.pedestrians = Pedestrians()
        self.lidar_pedestrians = Pedestrians()

        self.pub_pedestrians = rospy.Publisher('/combined_pedestrians',Pedestrians,queue_size=1)

        self.sub_pedestrians = rospy.Subscriber('/pedestrians',Pedestrians, self.cbPedestrians)
        self.sub_lidar_obstacles = rospy.Subscriber('/obstacles',Obstacles, self.cbLidarObstacles)

    def cbPedestrians(self, msg):
        self.pedestrians = msg

    def cbLidarObstacles(self, msg):
        pedestrians = Pedestrians()
        for obstacle in msg.circles:
            person = Person()
            person.position = obstacle.center

            person.velocity.x = obstacle.velocity.x
            person.velocity.y = obstacle.velocity.y
            person.velocity.z = obstacle.velocity.z

            pedestrians.radii.append(obstacle.true_radius)
            pedestrians.people.append(person)
            pedestrians.tags.append(0)

        self.lidar_pedestrians = pedestrians


    def publishPedestrian(self):
        combined_pedestrians = Pedestrians()
        combined_pedestrians.people.extend(self.pedestrians.people)
        combined_pedestrians.people.extend(self.lidar_pedestrians.people)
        combined_pedestrians.radii.extend(self.pedestrians.radii)
        combined_pedestrians.radii.extend(self.lidar_pedestrians.radii)
        combined_pedestrians.tags.extend(self.pedestrians.tags)
        combined_pedestrians.tags.extend(self.lidar_pedestrians.tags)
        self.pub_pedestrians.publish(combined_pedestrians)

def run():
    rospy.init_node('pedestrian_publisher',anonymous=False)
    pedestrian_publisher = PedestrianPublisher()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pedestrian_publisher.publishPedestrian();
        rate.sleep();

if __name__ == '__main__':
    run()
