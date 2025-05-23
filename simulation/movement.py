#!/usr/bin/env python

import rospy
import csv
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from std_srvs.srv import Empty
import time

class MovementCollector:
    def __init__(self, robot_name="robot", output_csv="movement_results.csv"):
        rospy.init_node('movement_collector', anonymous=True)

        self.cmd_vel_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
        rospy.wait_for_service('/gazebo/reset_world')
        rospy.wait_for_service('/gazebo/get_model_state')

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.samples_per_duration = 50
        self.durations = [2.5, 5.0, 7.5]  # in seconds
        self.robot_name = robot_name
        self.output_csv = output_csv

    def move_robot(self, vx=0.0, vy=0.0, omega=0.0, duration=1.0):
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        twist.angular.z = omega

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            self.cmd_vel_pub.publish(twist)

        # Stop the robot
        self.stop_robot()

    def stop_robot(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(vel_msg)

    def get_position(self):
        try:
            resp = self.get_model_state(self.robot_name, "")
            position = resp.pose.position
            rospy.loginfo(position)
            return position
        except:
            rospy.logerr("Could not get model state")
            return None

    def reset(self):
        try:
            self.reset_world()
        except:
            rospy.logerr("Failed to reset world")
        rospy.sleep(1)

    def run(self, vx=0.02, vy=0.0, omega=0.0):
        with open(self.output_csv, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Duration_s', 'Sample', 'Final_X', 'Final_Y', 'Final_Z'])

            for duration in self.durations:
                for sample in range(1, self.samples_per_duration + 1):
                    rospy.loginfo(f"Running sample {sample}/50 for duration {duration}s")
                    self.move_robot(vx=vx, vy=vy, omega=omega, duration=duration)
                    pos = self.get_position()
                    if pos:
                        writer.writerow([duration, sample, pos.x, pos.y, pos.z])
                    self.reset()

if __name__ == '__main__':
    try:
        collector = MovementCollector()
        # Run in x-direction (default)
        collector.run(vx=0.02, vy=0.0, omega=0.0)
        # collector.move_robot(vx=0.02, vy=0.0, omega=0.0, duration=10)
    except rospy.ROSInterruptException:
        pass
