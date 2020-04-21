from __future__ import print_function
import time
import numpy as np
import torch
import robot
import soft_actor_critic.sac_vrep as sac_vrep
import soft_actor_critic.vrep as vrep

class Visualiser:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot = robot.Vrep_Communication()
        self.policy = sac_vrep.PolicyNetwork().to(self.device)
        self.policy.load_state_dict(torch.load('/homes/gt4118/Desktop/Robot_Learning/sac.pt'))
        self.policy.eval()
        self.robot.establish_communication()
        self.robot.start_vrep()
        self.robot.initialise()
        self.robot.pick_color()
        self.robot.add_object()
        self.robot.reset_object_position_and_orientation()
        self.robot.get_initial_position()
        self.action_x_lower_bounds = -0.05
        self.action_x_upper_bounds = 0.05
        self.action_y_lower_bounds = -0.05
        self.action_y_upper_bounds = 0.05
        self.action_z_lower_bounds = -0.05
        self.action_z_upper_bounds = 0.05
        self.termination_height = 0.13
        self.hand_status = None

    def visualise(self):
        for j in range(1, 100):
            self.robot.set_initial_position()
            image = self.robot.get_image()
            height = 0.12765
            self.hand_status = 1
            numerical_state = np.array([self.hand_status, height])
            episode_reward = 0
            for step in range(30):
                img_tensor = torch.from_numpy(image)
                numerical_state_tensor = torch.from_numpy(numerical_state)
                action = self.policy.get_action(img_tensor.unsqueeze(0).permute(0, 3, 1, 2).\
                    float().to(self.device), numerical_state_tensor.\
                        unsqueeze(0).float().to(self.device))
                if action[0] < self.action_x_lower_bounds:
                    action[0] = self.action_x_lower_bounds
                elif action[0] > self.action_x_upper_bounds:
                    action[0] = self.action_x_upper_bounds
                if action[1] > self.action_y_upper_bounds:
                    action[1] = self.action_y_upper_bounds
                elif action[1] < self.action_y_lower_bounds:
                    action[1] = self.action_y_lower_bounds
                if action[2] > self.action_z_upper_bounds:
                    action[2] = self.action_z_upper_bounds
                elif action[2] < self.action_z_lower_bounds:
                    action[2] = self.action_z_lower_bounds
                if action[3] < 0.5:
                    action[3] = 0
                elif action[3] >= 0.5:
                    action[3] = 1

                _, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.\
                    client_id, self.robot.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
                old_sawyer_position = sawyer_target_position

                if sawyer_target_position[0] + action[0] < 1.0112: #the action now is 0
                    action[0] = 0.0001
                elif sawyer_target_position[0] + action[0] > 1.2412:
                    action[0] = 0.0001
                if sawyer_target_position[1] + action[1] < 0.95289: #the action now is 0
                    action[1] = 0.0001
                elif sawyer_target_position[1] + action[1] > 1.3409:
                    action[1] = 0.0001
                if sawyer_target_position[2] + action[2] < 0.027648: #the action now is 0
                    action[2] = 0.0001
                elif sawyer_target_position[2] + action[2] > 0.12765:
                    action[2] = 0.0001

                move_direction = np.asarray([action[0], action[1], action[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.01 * move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.01))
                remaining_magnitude = -num_move_steps * 0.01 + move_magnitude
                remaining_distance = remaining_magnitude * move_direction/move_magnitude
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.robot.client_id, self.robot.\
                        Sawyer_target_handle, -1, (sawyer_target_position[0] + move_step[0],\
                        sawyer_target_position[1] + move_step[1], sawyer_target_position[2] + \
                        move_step[2]), vrep.simx_opmode_blocking)
                    _, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.\
                        client_id, self.robot.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
                    vrep.simxSynchronousTrigger(self.robot.client_id)
                    vrep.simxGetPingTime(self.robot.client_id)
                vrep.simxSetObjectPosition(self.robot.client_id, self.robot.Sawyer_target_handle, \
                    -1, (sawyer_target_position[0] + remaining_distance[0], \
                    sawyer_target_position[1] + remaining_distance[1], sawyer_target_position[2] + \
                    remaining_distance[2]), vrep.simx_opmode_blocking)
                vrep.simxSynchronousTrigger(self.robot.client_id)
                vrep.simxGetPingTime(self.robot.client_id)

                if action[3] == 1: #open hand
                    since = time.time()
                    _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.\
                        motorHandle, vrep.simx_opmode_blocking)
                    vrep.simxSetJointForce(self.robot.client_id, self.robot.motorHandle, 20, \
                        vrep.simx_opmode_blocking)
                    vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motorHandle, \
                        -0.5, vrep.simx_opmode_blocking)
                    vrep.simxSynchronousTrigger(self.robot.client_id)
                    vrep.simxGetPingTime(self.robot.client_id)
                    while dist > -1e-06:
                        _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.\
                            motorHandle, vrep.simx_opmode_blocking)
                        vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.\
                            motorHandle, -0.5, vrep.simx_opmode_blocking)
                        vrep.simxSynchronousTrigger(self.robot.client_id)
                        vrep.simxGetPingTime(self.robot.client_id)

                    vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.\
                        motorHandle, 0.0, vrep.simx_opmode_blocking)
                    vrep.simxSynchronousTrigger(self.robot.client_id)
                    vrep.simxGetPingTime(self.robot.client_id)
                    self.hand_status = 1

                elif action[3] == 0:
                    since = time.time()
                    _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.\
                        motorHandle, vrep.simx_opmode_blocking)
                    vrep.simxSetJointForce(self.robot.client_id, self.robot.motorHandle, 100, \
                        vrep.simx_opmode_blocking)
                    vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.\
                        motorHandle, 0.5, vrep.simx_opmode_blocking)
                    vrep.simxSynchronousTrigger(self.robot.client_id)
                    vrep.simxGetPingTime(self.robot.client_id)
                    while dist < 0.0345:
                        _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.\
                            robot.motorHandle, vrep.simx_opmode_blocking)
                        vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.\
                            motorHandle, 0.5, vrep.simx_opmode_blocking)
                        vrep.simxSynchronousTrigger(self.robot.client_id)
                        vrep.simxGetPingTime(self.robot.client_id)
                        if time.time()-since > 5:
                            if self.robot.has_object():
                                break
                            print(dist)
                            print('trouble closing the gripper')
                    vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motorHandle,\
                        0.5, vrep.simx_opmode_blocking)
                    vrep.simxSynchronousTrigger(self.robot.client_id)
                    vrep.simxGetPingTime(self.robot.client_id)
                    self.hand_status = 0

                next_image = self.robot.get_image()
                _, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.client_id,\
                    self.robot.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
                next_numerical_state = np.array([self.hand_status, sawyer_target_position[2]])
                new_sawyer_position = sawyer_target_position
                _, object_position = vrep.simxGetObjectPosition(self.robot.client_id, \
                    self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
                done = False
                reward = -0.1
                success = 0
                if new_sawyer_position[2] > self.termination_height or not \
                    self.robot.check_object_inbounds():
                    done = True

                elif self.hand_status == 1 and action[3] == 1: #when moving towards the object
                    before = np.array(old_sawyer_position) - np.array(object_position)
                    after = np.array(new_sawyer_position) - np.array(object_position)
                    old_distance = np.linalg.norm(before) #5
                    new_distance = np.linalg.norm(after) #3
                    reward += old_distance-new_distance #incentivize to move closer to the object

                elif self.hand_status == 0:
                    distance = np.linalg.norm(np.array(object_position) - \
                        np.array(new_sawyer_position))
                    hold = self.robot.has_object()
                    if hold: #holds the object
                        self.robot.lift_arm()
                        label = self.robot.successful_grasp()
                        if label == 0:
                            done = True
                            reward += -distance
                        elif label == 1:
                            reward += 10 #reward it
                            done = True
                            success = 1
                    else:
                        reward += -distance

                if done:
                    next_image = None
                    next_numerical_state = None

                image = next_image
                numerical_state = next_numerical_state
                episode_reward += reward

                if done:
                    break
            print(episode_reward)
            self.robot.reset_object_position_and_orientation()

LEARNEDPOLICY = Visualiser()
LEARNEDPOLICY.visualise()
