import random
import time
import numpy as np
import sys
from gym import spaces
import gym

import os
import math 
import pybullet as p
import pybullet_data
from datetime import datetime
from collections import namedtuple
from attrdict import AttrDict

PROJECT_DATA = "./ur_e_description/urdf"
ROBOT_URDF_PATH = os.path.join(PROJECT_DATA , "robot/ur5_robotiq_85.urdf")
TABLE_URDF_PATH = os.path.join(PROJECT_DATA , "table/table.urdf")
CUBE_URDF_PATH = os.path.join(PROJECT_DATA , "cube/red.urdf")
TRAY_URDF_PATH = [os.path.join(PROJECT_DATA , f"tray/tray_{color}.urdf") for color in ['red', 'green', 'blue']]
# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

class Tray:
    """
    Tray configuration for sorting objects by color
    """
    def __init__(self, uid, color, position):
        self.uid = uid
        self.color = color  # Color of the tray
        self.pos = position  # Fixed position of the tray

class ur5GymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 useIK=True,
                 actionRepeat=120,
                 renders=True,
                 maxSteps=200,
                 simulatedGripper=False,
                 randObjPos=False,
                 task=4, # here target number
                 learning_param=0):

        self.renders = renders
        self.actionRepeat = actionRepeat
        self.obj_color = 'red'
        self.useIK = useIK
        # setup pybullet sim:
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setTimeStep(1./240.)
        p.setGravity(0,0,-10)
        p.setRealTimeSimulation(False)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
        # p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0,0,0])
        p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0,0,0])

        
        
        
        # setup robot arm:
        self.end_effector_index = 7
        flags = p.URDF_USE_SELF_COLLISION
        self.ur5 = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = p.getNumJoints(self.ur5)
        # print(self.num_joints)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = p.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointType != 'FIXED' else False
            # controllable = True if jointName in self.mimic_children_names else controllable
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            # print(info)
            if controllable:
                p.setJointMotorControl2(self.ur5, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
        # for joint_name, joint_info in self.joints.items():
        #     print(f"Joint Name: {joint_name}")
        #     print(f"Joint Info: {joint_info}")

        # gripper info:
        self.mimic_parent_name = 'finger_joint'
        # self.mimic_children_names = ['right_outer_knuckle_joint',
        #                         'left_inner_knuckle_joint',
        #                         'right_inner_knuckle_joint',
        #                         'left_inner_finger_joint',
        #                         'right_inner_finger_joint']
        # self.mimic_multiplier = [1, 1, 1, -1, -1]
        self.mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        
        self.setup_mimic_joints()
        #table
        self.table = p.loadURDF(TABLE_URDF_PATH, [0.45, -0.1, -0.65], [0, 0, 0, 1])
        
        # Tray
        new_tray_positions = [[0.25, -0.4, 0], [0.5, -0.4, 0], [0.75, -0.4, 0]]  # Positions for new trays
        new_tray_colors = ["green", "red", "blue"]
        
        new_tray_uids = [
            p.loadURDF(os.path.join(PROJECT_DATA , f"tray/tray_{color}.urdf"), basePosition=pos)
            for pos, color in zip(new_tray_positions, new_tray_colors)
        ]
        trays = [Tray(uid, color, pos) for uid, color, pos in zip(new_tray_uids, new_tray_colors, new_tray_positions)]
        # object:
        self.initial_obj_pos = [0.8, 0.1, 0] # initial object pos
        for tray in trays:
            if tray.color == self.obj_color:
                self.initial_target_pos = tray.pos
                # self.initial_target_pos[2] = self.initial_target_pos[2]+0.1
                # print(self.initial_target_pos)
                # print(self.initial_obj_pos)
                break
        self.obj = p.loadURDF(CUBE_URDF_PATH, self.initial_obj_pos)
        
        
        if self.useIK:
            self.action_dim = 4 # IK: 3 coordinates, 1 gripper
        else:
            self.action_dim = 7 # direct control: 6 arm DOF, 1 gripper
            
        self.maxSteps = maxSteps
        # self.randObjPos = randObjPos
        self.observation = np.array(0)

        self.task = task
        self.learning_param = learning_param
     
        self._action_bound = 1.0 # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
        high = np.array([10]*self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')
        
    def reset(self):
        self.stepCounter = 0
        self.current_task = 0
        self.obj_picked_up = False
        self.ur5_or = [0.0, 1/2*math.pi, 0.0]
        self.target_pos = self.initial_target_pos

        # pybullet.addUserDebugText('X', self.obj_pos, [0,1,0], 1) # display goal
        # if self.randObjPos:
        # self.initial_obj_pos = [0.6+random.random()*0.1, 0.1+random.random()*0.1, 0.0]
        p.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0.,0.,0.,1.0]) # reset object pos

        # reset robot simulation and position:
        joint_angles = (-0.34, -1.57, 1.80, -1.57, -1.57, 0.00) # pi/2 = 1.5707
        if len(joint_angles) != len(self.control_joints):
            raise ValueError("joint_angles length does not match control_joints length.")
        self.set_joint_angles(joint_angles)

        # reset gripper:
        open_length = 0.085
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143) 
        self.control_gripper(open_angle) # open

        # step simualator:
        for i in range(10):
            self.step_simulation()

        # get obs and return:
        self.getExtendedObservation()
        return self.observation
    
    def step(self, action):
        action = np.array(action)
        arm_action = action[0:self.action_dim-1].astype(float) # dX, dY, dZ - range: [-1,1]
        gripper_action = action[self.action_dim-1].astype(float) # gripper - range: [-1=closed,1=open]
        # print(gripper_action)
        # simualted gripping:
        # if self.obj_picked_up and gripper_action < -0.1:
        if self.current_task == 1 or self.current_task == 2:
            # object follows the arm tool tip:
            object_pos = self.get_current_pose()[0] # XYZ, no angles
            p.resetBasePositionAndOrientation(self.obj, object_pos, [0.,0.,0.,1.0])

        # get current position:
        if self.useIK:
            cur_p = self.get_current_pose()
        else:
            cur_p = self.get_joint_angles()

        # add delta position:
        new_p = np.array(cur_p[0]) + arm_action
        # print(new_p)
        # actuate:
        if self.useIK:
            joint_angles = self.calculate_ik(new_p, self.ur5_or) # XYZ and angles set to zero
        else:
            joint_angles = new_p
        # print(joint_angles)
        # for i in range(100):
        #     self.step_simulation()
        self.set_joint_angles(joint_angles[:6])

        # operate gripper: close = 0.8, open = 0, 2.5 to scale to std=1, nn max value
        gripper_action = np.clip(gripper_action/2.5, -0.4, 0.4) + 0.4
        # print(gripper_action)
        self.control_gripper(gripper_action)

        
        # step simualator:
        for i in range(self.actionRepeat):
            self.step_simulation()
        
        self.getExtendedObservation()
        reward = self.compute_reward() # call this after getting obs!
        done = self.my_task_done()

        info = {}
        # info = {'is_success': False}
        # if self.terminated == self.task:
        #     info['is_success'] = True

        self.stepCounter += 1

        return self.observation, reward, done, info
    
    def compute_reward(self):
        reward = np.zeros(1)

        self.target_dist = goal_distance(np.array(self.tool_pos), 
                                        np.array(self.goal_pos))
        # print(self.target_dist)

        # check approach velocity:
        # tv = self.tool.getVelocity()
        # approach_velocity = np.sum(tv)

        # print(approach_velocity)
        # input()

        reward += -self.target_dist * 10

        # task 0,2: reach object/target:
        # print(self.current_task)
        
        if self.current_task == 0 or self.current_task == 2:
            if self.target_dist < self.learning_param:# and approach_velocity < 0.05:
                if self.current_task == 0:
                    self.obj_picked_up = True
                    print('Successful object reach')
                if self.current_task == 2:
                    self.obj_picked_up = False
                    print('Successful target reach')
                self.current_task += 1
        
        # task 1,3: lift up:
        if self.current_task == 1 or self.current_task == 3:
            if self.tool_pos[2] > 0.3:# and approach_velocity < 0.05:
                if self.current_task == 1:
                    print('Successful picked up!')
                if self.current_task == 3:
                    print('Successful drop!')
                self.current_task += 1
                print('Successful!')

        # penalize if it tries to go lower than desk / platform collision:
        if self.tool_pos[2] < 0: # lower than position of object!
            reward += -1
            # print('Penalty: lower than desk!')

        # check collisions:
        if self.check_collisions(): 
            reward += -1
            # print('Collision!')

        # print(target_dist, reward)
        # input()

        return reward
        
    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        if self.current_task == self.task:
            print(f'done with {self.current_task}')
        c = (self.current_task == self.task or self.stepCounter > self.maxSteps)
        return c
    
    def getExtendedObservation(self):
        # sensor values:
        # js = self.get_joint_angles()
        self.tool_pos,_ = self.get_current_pose() # XYZ, no angles
        self.obj_pos,_ = p.getBasePositionAndOrientation(self.obj)
        self.observation = np.array(np.concatenate((self.tool_pos, self.obj_pos)))

        # we define tasks as: 0-reach obj, 1-lift ojb, 2-move to target, 3-drop obj
        self.goal_pos = self.obj_pos
        if self.current_task == 2: # reach target pos
            self.goal_pos = self.target_pos
            p.addUserDebugText('X', self.goal_pos, [0,1,0], 1) # display goal
            
    def check_collisions(self):
        collisions = p.getContactPoints()
        if len(collisions) > 0:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False
    
    def control_gripper(self, gripper_opening_angle):
        p.setJointMotorControl2(
            self.ur5,
            self.joints[self.mimic_parent_name].id,
            p.POSITION_CONTROL,
            targetPosition=gripper_opening_angle,
            force=self.joints[self.mimic_parent_name].maxForce,
            maxVelocity=self.joints[self.mimic_parent_name].maxVelocity)
        # for i in range(100):
        #     self.step_simulation()
        # for i in range(len(self.mimic_children_names)):
        #     joint = self.joints[self.mimic_children_names[i]]
        #     p.setJointMotorControl2(
        #         self.ur5, joint.id, p.POSITION_CONTROL,
        #         targetPosition=gripper_opening_angle * self.mimic_multiplier[i],
        #         force=joint.maxForce,
        #         maxVelocity=joint.maxVelocity)
            
    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        # print("Poses:", poses)
        # print("Indexes:", indexes)
        # print("Forces:", forces)
        # print('joint angle: ', joint_angles)

        if len(joint_angles) != len(indexes):
            raise ValueError("Mismatch between poses and joint indexes. Check your control_joints and joint_angles.")
        # self.step_simulation()
        p.setJointMotorControlArray(
            self.ur5, indexes,
            p.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )
    def get_joint_angles(self):
        j = p.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    
    def get_current_pose(self):
        linkstate = p.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
    
    def calculate_ik(self, position, orientation):
        quaternion = p.getQuaternionFromEuler(orientation)
        # print(quaternion)
        # quaternion = (0,1,0,1)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        # rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
        rest_poses = [(-0.34, -1.57, 1.80, -1.57, -1.57, 0.00)] # rest pose of our ur5 robot

        joint_angles = p.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*self.num_joints, 
            upperLimits=upper_limits, 
            lowerLimits=lower_limits, 
            jointRanges=joint_ranges, 
            restPoses=rest_poses,
            maxNumIterations=20
        )
        return joint_angles
    def step_simulation(self):
        p.stepSimulation()
        if self.render:
            time.sleep(1./240.)
            
    def setup_mimic_joints(self):
        self.mimic_parent_id = [joint_info.id for _,joint_info in self.joints.items() if joint_info.name == self.mimic_parent_name][0]
        # print(self.mimic_parent_id)
        self.mimic_child_multiplier = {joint_info.id: self.mimic_children_names[joint_info.name] for _,joint_info in self.joints.items() if joint_info.name in self.mimic_children_names}
        # print(self.mimic_child_multiplier)
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            # print(joint_id, multiplier)
            c = p.createConstraint(self.ur5, self.mimic_parent_id,
                                self.ur5, joint_id,
                                jointType=p.JOINT_GEAR,
                                jointAxis=[0, 1, 0],
                                parentFramePosition=[0, 0, 0],
                                childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=0.5)  # Note: the mysterious `erp` is of EXTREME importance
            # print(f"Constraint ID: {c}")
    
            # # Lấy thông tin ràng buộc
            # constraint_info = p.getConstraintInfo(c)
            # print(f"Constraint Info for ID {c}: {constraint_info}")
def main():
    env = ur5GymEnv(renders=True)
    env.reset()
    for i in range(100):
        open_length = float(input())
        # while True: # print(f"Lần {i + 1}: Mở tay gắp")
        
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143) 
        print(open_angle)
        # opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        # # angle calculation
        env.control_gripper(open_angle)
        for _ in range(120):
            env.step_simulation()
              # Đóng tay gắp
        # time.sleep(1/240)  # Đợi 1 giây để quan sát
        # p.stepSimulation()
        # print( opening_angle)
        print("Hoàn thành việc đóng mở tay gắp 5 lần.")
    
if __name__ == "__main__":
    main()