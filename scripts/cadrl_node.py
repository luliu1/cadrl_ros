#!/usr/bin/env python

import rospy
import sys
import configparser
from std_msgs.msg import Float32, ColorRGBA, Int32, UInt8
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point
from navigation_msgs.msg import Pedestrians, PlannerMode
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from crowd_nav_policy import SARL
from crowd_sim_utils import FullState, ObservableState, JointState, ActionXY

import numpy as np
import numpy.matlib
from matplotlib import cm
import matplotlib.pyplot as plt
import copy
import os
import logging

import rospkg
from tf.transformations import euler_from_quaternion
import tf
import torch

import network
import agent
import util

PED_RADIUS = 0.3
# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

class NN_jackal():
    def __init__(self, veh_name, veh_data, nn, actions):
        self.node_name = rospy.get_name()
        self.prev_other_agents_state = []

        # vehicle info
        self.veh_name = veh_name
        self.veh_data = veh_data

        # self.agent = agent.Agent(0.0, 0.0, 100.0, 100.0, radius, pref_speed, initial_heading, id)

        # neural network
        self.nn = nn
        self.actions = actions
        # self.value_net = value_net
        self.operation_mode = PlannerMode()
        self.operation_mode.mode = self.operation_mode.NN
        self.time_step = 0.25

        # for subscribers
        self.pose = PoseStamped()
        self.vel = Vector3(0.0, 0.0, 0.0)
        self.psi = 0.0
        self.ped_traj_vec = []
        self.other_agents_state = []

        # for publishers
        self.global_goal = PoseStamped()
        self.goal = PoseStamped()
        self.goal.pose.position.x = veh_data['goal'][0]
        self.goal.pose.position.y = veh_data['goal'][1]
        self.desired_position = PoseStamped()
        self.desired_action = np.zeros((2,))

        # handle obstacles close to vehicle's front
        self.stop_moving_flag = False
        self.d_min = 0.0
        self.new_subgoal_received = False
        self.new_global_goal_received = False
        self.safety_counter = 0;
        self.goal_counter = 0;
        self.time_goal_received = 0;

        # visualization
        self.path_marker = Marker()

        # subscribers and publishers
        self.num_poses = 0
        self.num_actions_computed = 0.0
        self.pub_others = rospy.Publisher('~other_vels',Vector3,queue_size=1)
        self.pub_twist = rospy.Publisher('~nn_cmd_vel',Twist,queue_size=1)
        self.pub_pose_marker = rospy.Publisher('~pose_marker',Marker,queue_size=1)
        self.pub_agent_marker = rospy.Publisher('~agent_marker',Marker,queue_size=1)
        self.pub_agent_markers = rospy.Publisher('~agent_markers',MarkerArray,queue_size=1)
        self.pub_path_marker = rospy.Publisher('~path_marker',Marker,queue_size=1)
        self.pub_goal_path_marker = rospy.Publisher('~goal_path_marker',Marker,queue_size=1)
        self.sub_pose = rospy.Subscriber('~pose',TFMessage,self.cbPose)
        self.sub_vel = rospy.Subscriber('~wheel_odo',Odometry,self.cbVel)
        self.sub_mode = rospy.Subscriber('~mode',PlannerMode, self.cbPlannerMode)
        self.sub_global_goal = rospy.Subscriber('~goal',PoseStamped, self.cbGlobalGoal)
        self.sub_safety_counter = rospy.Subscriber('/control_safety_level', UInt8, self.cbSafetyCounter)

        self.sub_pedestrians = rospy.Subscriber('/combined_pedestrians',Pedestrians, self.cbPedestrians)

        # control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.01),self.cbControl)
        self.nn_timer = rospy.Timer(rospy.Duration(0.1),self.cbComputeActionGA3C)
        self.transform_listener = tf.TransformListener()

        #logging
        self.logger = logging.getLogger('Collisions')
        hdlr = logging.FileHandler('cadrl_collisions.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Logging the number of potential collisions & time per path.')

        #logging
        self.loggerTimings = logging.getLogger('Timings')
        handler = logging.FileHandler('cadrl_timings.log')
        hdlr.setFormatter(formatter)
        self.loggerTimings.addHandler(handler)
        self.loggerTimings.setLevel(logging.DEBUG)
        self.loggerTimings.debug('Logging the timings of the network.')

    def cbGlobalGoal(self,msg):
        self.new_global_goal_received = True
        self.goal_counter += 1
        self.safety_counter = 0
        self.time_goal_received = rospy.get_time()
        self.global_goal = msg
        self.operation_mode.mode = self.operation_mode.SPIN_IN_PLACE

        self.goal.pose.position.x = msg.pose.position.x
        self.goal.pose.position.y = msg.pose.position.y
        self.goal.header = msg.header
        self.new_subgoal_received = True

    def cbSafetyCounter(self,msg):
        if msg.data > 0:
            self.safety_counter += 1

    def cbPlannerMode(self, msg):
        self.operation_mode = msg
        self.operation_mode.mode = self.operation_mode.NN

    def cbPose(self, msg):
        self.num_poses += 1
        (translation, rotation) = self.transform_listener.lookupTransform("G_ground_frame", "R_robot_base_frame", rospy.Time(0))
        euler =  euler_from_quaternion(rotation)
        self.psi = euler[2]
        self.pose.pose.position.x = translation[0]
        self.pose.pose.position.y = translation[1]
        self.pose.pose.position.z = translation[2]
        self.pose.pose.orientation.x = rotation[0]
        self.pose.pose.orientation.y = rotation[1]
        self.pose.pose.orientation.z = rotation[2]
        self.pose.pose.orientation.w = rotation[3]

        #self.goal.pose.position.x = self.pose.pose.position.x + 4.0 * np.cos(self.psi)
        #self.goal.pose.position.y = self.pose.pose.position.y + 4.0 * np.sin(self.psi)
        self.visualize_pose(self.pose.pose.position,self.pose.pose.orientation)

    def cbVel(self, msg):
        vel_ego = msg.twist.twist.linear
        speed = np.sqrt(np.square(vel_ego.x) + np.square(vel_ego.y))
        self.vel.x = speed * np.cos(self.psi)
        self.vel.y = speed * np.sin(self.psi)

    def cbPedestrians(self, msg):
        other_agents = []

        xs = []; ys = []; radii = []; labels = []; v_xs = []; v_ys = [];
        num_pedestrians = len(msg.people)
        for i in range(num_pedestrians):
            index = i
            x = msg.people[i].position.x
            y = msg.people[i].position.y
            v_x = msg.people[i].velocity.x
            v_y = msg.people[i].velocity.y
            radius = PED_RADIUS#max(PED_RADIUS, msg.radii[i])

            xs.append(x); ys.append(y); radii.append(radius);
            labels.append(index); v_xs.append(v_x); v_ys.append(v_y);
            # self.visualize_other_agent(x,y,radius,msg.labels[i])
            # helper fields
            heading_angle = np.arctan2(v_y, v_x)
            pref_speed = np.linalg.norm(np.array([v_x, v_y]))
            goal_x = x + 5.0; goal_y = y + 5.0

            if pref_speed < 0.2:
                pref_speed = 0; v_x = 0; v_y = 0
            other_agents.append(agent.Agent(x, y, goal_x, goal_y, radius, pref_speed, heading_angle, index))
            other_agents[i].vel_global_frame = np.array([v_x, v_y])
        if num_pedestrians > 0:
            self.visualize_other_agents(xs, ys, radii, labels, v_xs, v_ys)
        self.other_agents_state = other_agents

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)
        # print 'Stop Moving.'

    def update_action(self, action):
        # print 'update action'
        self.desired_action = action
        self.desired_position.pose.position.x = self.pose.pose.position.x + 1*action[0]*np.cos(action[1])
        self.desired_position.pose.position.y = self.pose.pose.position.y + 1*action[0]*np.sin(action[1])

        twist = Twist()
        twist.linear.x = action[0]
        yaw_error = action[1] - self.psi
        if yaw_error > np.pi:
            yaw_error -= 2*np.pi
        if yaw_error < -np.pi:
            yaw_error += 2*np.pi
        twist.angular.z = yaw_error

    def find_vmax(self, d_min, heading_diff):
        # Calculate maximum linear velocity, as a function of error in
        # heading and clear space in front of the vehicle
        # (With nothing in front of vehicle, it's not important to
        # track MPs perfectly; with an obstacle right in front, the
        # vehicle must turn in place, then drive forward.)
        d_min = max(0.0,d_min)
        x = 0.3
        margin = 0.3
        # y = max(d_min - 0.3, 0.0)
        y = max(d_min, 0.0)
        # making sure x < y
        if x > y:
            x = 0
        w_max = 1
        # x^2 + y^2 = (v_max/w_max)^2
        v_max = w_max * np.sqrt(x**2 + y**2)
        v_max = np.clip(v_max,0.0,self.veh_data['pref_speed'])
        # print 'V_max, x, y, d_min', v_max, x, y, d_min
        if abs(heading_diff) < np.pi / 18:
            return self.veh_data['pref_speed']
        return v_max

    def cbControl(self, event):
        #if self.goal.header.stamp == rospy.Time(0) or self.stop_moving_flag \
        #    and not self.new_global_goal_received:
        #    self.stop_moving()
        #    return
        if self.operation_mode.mode==self.operation_mode.NN:
            desired_yaw = self.desired_action[1]
            yaw_error = desired_yaw - self.psi
            if abs(yaw_error) > np.pi:
                yaw_error -= np.sign(yaw_error)*2*np.pi
            gain = 1
            vw = gain*yaw_error

            use_d_min = False
            if use_d_min:
                use_d_min = True
                vx = min(self.desired_action[0], self.find_vmax(self.d_min,yaw_error))
            else:
                vx = self.desired_action[0]

            twist = Twist()
            twist.angular.z = vw
            twist.linear.x = vx
            self.pub_twist.publish(twist)
            self.visualize_action(use_d_min)
            return

        if self.operation_mode.mode == self.operation_mode.SPIN_IN_PLACE:
            print ('Spinning in place.')
            self.stop_moving_flag = False
            angle_to_goal = np.arctan2(self.global_goal.pose.position.y - self.pose.pose.position.y, \
                self.global_goal.pose.position.x - self.pose.pose.position.x)
            global_yaw_error = self.psi - angle_to_goal
            if abs(global_yaw_error) > 0.5:
                vx = 0.0
                vw = 1.0
                twist = Twist()
                twist.angular.z = vw
                twist.linear.x = vx
                self.pub_twist.publish(twist)
            else:
                print ('Done spinning in place')
                self.operation_mode.mode = self.operation_mode.NN
                self.new_global_goal_received = False
            return
        else:
            self.stop_moving()
            return

    def cbComputeActionSARL(self, event):
        startTime = rospy.Time.now();
        occupancy_maps = None
        probability = np.random.random()
        pref_speed = self.veh_data['pref_speed']

        action_values = list()
        max_value = float('-inf')
        max_action = None

        x = self.pose.pose.position.x; y = self.pose.pose.position.y
        v_x = self.vel.x; v_y = self.vel.y
        radius = self.veh_data['radius']; turning_dir = 0.0
        heading_angle = self.psi
        pref_speed = self.veh_data['pref_speed']
        goal_x = self.goal.pose.position.x; goal_y = self.goal.pose.position.y
        # in case current speed is larger than desired speed
        v = np.linalg.norm(np.array([v_x, v_y]))
        if v > pref_speed:
            v_x = v_x * pref_speed / v
            v_y = v_y * pref_speed / v

        full_state = FullState(x, y, v_x, v_y, radius, goal_x, goal_y, pref_speed, heading_angle)
        ob = [ObservableState(other_agent.pos_global_frame[0], other_agent.pos_global_frame[1], other_agent.vel_global_frame[0], other_agent.vel_global_frame[1], other_agent.radius) for other_agent in self.other_agents_state]
        state = JointState(full_state, ob)
        if len(state.human_states) > 0:
            for action in self.nn.action_space:
                next_self_state = self.nn.propagate(state.self_state, action)
                next_human_states = [self.nn.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                   for human_state in state.human_states]
                reward = self.nn.compute_reward(next_self_state, next_human_states)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.nn.device)
                                              for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.nn.rotate(batch_next_states).unsqueeze(0)
                if self.nn.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.nn.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE
                next_state_value = self.nn.model(rotated_batch_input).data.item()
                value = reward + pow(self.nn.gamma, self.time_step * pref_speed) * next_state_value
                action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
                # print ("People detected", max_action)
            max_action = np.array([max_action[0], util.wrap(max_action[1] + self.psi)])
        else:
            max_action = np.zeros((2,))
            max_action[0] = pref_speed
            angle_to_goal= np.arctan2(goal_y - y, goal_x - x)
            max_action[1] = angle_to_goal
            # print(max_action)
        if max_action is None:
            raise ValueError('Value network is not well trained. ')
        timingNetwork = rospy.Time.now() - startTime
        self.loggerTimings.debug('%s', timingNetwork.nsecs)
        self.update_action(max_action)


    def cbComputeActionGA3C(self, event):
        startTime = rospy.Time.now();
        if self.operation_mode.mode!=self.operation_mode.NN:
            print ('Not in NN mode')
            print (self.operation_mode.mode)
            return
        # construct agent_state
        x = self.pose.pose.position.x; y = self.pose.pose.position.y
        v_x = self.vel.x; v_y = self.vel.y
        radius = self.veh_data['radius']; turning_dir = 0.0
        heading_angle = self.psi
        pref_speed = self.veh_data['pref_speed']
        goal_x = self.goal.pose.position.x; goal_y = self.goal.pose.position.y
        # in case current speed is larger than desired speed
        v = np.linalg.norm(np.array([v_x, v_y]))
        if v > pref_speed:
            v_x = v_x * pref_speed / v
            v_y = v_y * pref_speed / v

        host_agent = agent.Agent(x, y, goal_x, goal_y, radius, pref_speed, heading_angle, 0)
        host_agent.vel_global_frame = np.array([v_x, v_y])
        host_agent.update_state([0.0,0.0],0.0)
        #host_agent.print_agent_info()

        other_agents_state = copy.deepcopy(self.other_agents_state)
        obs = host_agent.observe(other_agents_state)[1:]
        obs = np.expand_dims(obs, axis=0)
        #print ("obs:", obs)
        predictions = self.nn.predict_p(obs, None)[0]
        #print ("predictions:", predictions)
        # print "best action index:", np.argmax(predictions)
        raw_action = copy.deepcopy(self.actions[np.argmax(predictions)])
        action = np.array([pref_speed*raw_action[0], util.wrap(raw_action[1] + self.psi)])
        # print "raw_action:", raw_action
        # print "action:", action
        # if close to goal
        kp_v = 0.5
        kp_r = 1
        if host_agent.dist_to_goal < 2.0: # and self.percentComplete>=0.9:
            # print "somewhat close to goal"
            pref_speed = max(min(kp_v * (host_agent.dist_to_goal-0.1), pref_speed), 0.0)
            action[0] = min(raw_action[0], pref_speed)
            turn_amount = max(min(kp_r * (host_agent.dist_to_goal-0.1), 1.0), 0.0) * raw_action[1]
            action[1] = util.wrap(turn_amount + self.psi)
        if host_agent.dist_to_goal < 0.3:
            if self.goal_counter > 0 and not self.stop_moving_flag:
                self.logger.debug('Number of collisions for run %s: %s', self.goal_counter, self.safety_counter)
                self.logger.debug('Time to get to goal: %s', rospy.get_time()-self.time_goal_received)
            self.stop_moving_flag = True
        else:
            self.stop_moving_flag = False

        # print 'chosen action (rel angle)', action[0], action[1]
        timingNetwork = rospy.Time.now() - startTime
        self.loggerTimings.debug('%s', timingNetwork.nsecs)
        self.update_action(action)

    def update_subgoal(self,subgoal):
        self.goal.pose.position.x = subgoal[0]
        self.goal.pose.position.y = subgoal[1]

    def visualize_subgoal(self,subgoal, subgoal_options=None):
        markers = MarkerArray()

        # Display GREEN DOT at NN subgoal
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'G_ground_frame'
        marker.ns = 'subgoal'
        marker.id = 0
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position.x = subgoal[0]
        marker.pose.position.y = subgoal[1]
        marker.scale = Vector3(x=0.4,y=0.4,z=0.2)
        marker.color = ColorRGBA(g=1.0,a=1.0)
        marker.lifetime = rospy.Duration(2.0)
        self.pub_goal_path_marker.publish(marker)

        if subgoal_options is not None:
            for i in xrange(len(subgoal_options)):
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = 'G_ground_frame'
                marker.ns = 'subgoal'
                marker.id = i+1
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.pose.position.x = subgoal_options[i][0]
                marker.pose.position.y = subgoal_options[i][1]
                marker.scale = Vector3(x=0.2,y=0.2,z=0.2)
                marker.color = ColorRGBA(b=1.0,r=1.0,a=1.0)
                marker.lifetime = rospy.Duration(1.0)
                self.pub_goal_path_marker.publish(marker)




    def visualize_pose(self,pos,orientation):
        # Yellow Box for Vehicle
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'G_ground_frame'
        marker.ns = 'agent'
        marker.id = 0
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = pos
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.5,y=0.3,z=0.1)
        marker.color = ColorRGBA(r=1.0,g=1.0,a=1.0)
        marker.lifetime = rospy.Duration(1.0)
        self.pub_pose_marker.publish(marker)

        # Red track for trajectory over time
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'G_ground_frame'
        marker.ns = 'agent'
        marker.id = self.num_poses
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = pos
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.2,y=0.2,z=0.2)
        marker.color = ColorRGBA(r=1.0,a=1.0)
        marker.lifetime = rospy.Duration(10.0)
        self.pub_pose_marker.publish(marker)

    def visualize_other_agents(self,xs,ys,radii,labels,vx, vy):
        markers = MarkerArray()
        for i in range(len(xs)):
            # Orange box for other agent
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = 'G_ground_frame'
            marker.ns = 'other_agent'
            marker.id = labels[i]
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.pose.position.x = xs[i]
            marker.pose.position.y = ys[i]
            # marker.pose.orientation = orientation
            marker.scale = Vector3(x=2*radii[i],y=2*radii[i],z=1)
            marker.color = ColorRGBA(r=1.0,g=0.4,a=1.0)
            marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(marker)

            # Display BLUE ARROW from current position to NN desired position
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = 'G_ground_frame'
            marker.ns = 'agent_path_arrow'
            marker.id = labels[i]
            marker.type = marker.ARROW
            marker.action = marker.ADD
            agent_position = Point()
            agent_position_end = Point()
            agent_position.x = xs[i]
            agent_position.y = ys[i]
            agent_position_end.x = xs[i] + vx[i]
            agent_position_end.y = ys[i] + vy[i]
            marker.points.append(agent_position)
            marker.points.append(agent_position_end)
            marker.scale = Vector3(x=0.1,y=0.2,z=0.2)
            marker.color = ColorRGBA(b=1.0,a=1.0)
            marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(marker)

        self.pub_agent_markers.publish(markers)

    def visualize_action(self, use_d_min):
        # Display BLUE ARROW from current position to NN desired position
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'G_ground_frame'
        marker.ns = 'path_arrow'
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.points.append(self.pose.pose.position)
        marker.points.append(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.1,y=0.2,z=0.2)
        marker.color = ColorRGBA(b=1.0,a=1.0)
        marker.lifetime = rospy.Duration(0.5)
        self.pub_goal_path_marker.publish(marker)

        # Display BLUE DOT at NN desired position
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'G_ground_frame'
        marker.ns = 'path_trail'
        marker.id = self.num_poses
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = copy.deepcopy(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.2,y=0.2,z=0.2)
        marker.color = ColorRGBA(b=1.0,a=0.1)
        marker.lifetime = rospy.Duration(0.5)
        if self.desired_action[0] == 0.0:
            marker.pose.position.x += 2.0*np.cos(self.desired_action[1])
            marker.pose.position.y += 2.0*np.sin(self.desired_action[1])
        self.pub_goal_path_marker.publish(marker)

        # Display RED LINE from along minimum clear distance in front
        # marker = Marker()
        # marker.header.stamp = rospy.Time.now()
        # marker.header.frame_id = 'odom'
        # marker.ns = 'clear_distance'
        # marker.id = 0
        # marker.type = marker.LINE_LIST
        # marker.lifetime = rospy.Duration(0.5)
        # marker.scale = Vector3(x=0.08,y=0.08,z=0.08)
        # if use_d_min:
        #     marker.color = ColorRGBA(r=1.0,a=1.0)
        # else:
        #     marker.color = ColorRGBA(r=1.0,g=1.0,a=1.0)
        # x_midpt = self.pose.pose.position.x + self.d_min*np.cos(self.psi)
        # y_midpt = self.pose.pose.position.y + self.d_min*np.sin(self.psi)
        # x_max = x_midpt - 1*np.sin(self.psi)
        # x_min = x_midpt + 1*np.sin(self.psi)
        # y_max = y_midpt + 1*np.cos(self.psi)
        # y_min = y_midpt - 1*np.cos(self.psi)
        # marker.points.append(Point(x=x_max,y=y_max))
        # marker.points.append(Point(x=x_min,y=y_min))
        # self.pub_goal_path_marker.publish(marker)


    def on_shutdown(self):
        rospy.loginfo("[%s] Shutting down." %(self.node_name))
        self.stop_moving()
        rospy.loginfo("Stopped %s's velocity." %(self.veh_name))


def run():
    print ('hello world from cadrl_node.py')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    plt.rcParams.update({'font.size': 18})
    rospack = rospkg.RosPack()

    a = network.Actions()
    actions = a.actions
    #HERE for CADRL
    num_actions = a.num_actions
    nn = network.NetworkVP_rnn(network.Config.DEVICE, 'network', num_actions)
    nn.simple_load(rospack.get_path('cadrl_ros')+'/checkpoints/network_01900000')
    #HERE for SARL
    #nn = SARL()
    #policy_config_file = '/home/lucia/catkin_ws/src/cadrl_ros/scripts/configs/policy.config'
    #policy_config = configparser.RawConfigParser()
    #policy_config.read(policy_config_file)
    #nn.configure(policy_config)
    #model_weights = os.path.join('/home/lucia/catkin_ws/src/cadrl_ros/src/crowd_nav/data/output_om', 'rl_model.pth')
    #if nn.trainable:
    #    nn.get_model().load_state_dict(torch.load(model_weights))
    #nn.set_phase('test')
    #device = torch.device("cpu")
    #nn.set_device(device)
    #TO HERE
    rospy.init_node('nn_jackal',anonymous=False)
    veh_name = 'JA01'
    pref_speed = rospy.get_param("~jackal_speed")
    #nn.build_action_space(pref_speed)
    veh_data = {'goal':np.zeros((2,)),'radius':0.5,'pref_speed':pref_speed,'kw':10.0,'kp':1.0,'name':'JA01'}

    print ("********\n*******\n*********\nJackal speed:", pref_speed, "\n**********\n******")

    nn_jackal = NN_jackal(veh_name, veh_data, nn, actions)
    rospy.on_shutdown(nn_jackal.on_shutdown)

    rospy.spin()

if __name__ == '__main__':
    run()
