from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import utils
import buffer
import vrep
plt.ion()

class Main:

    def __init__(self):
        self.batch_size = 128
        self.reward_scale = 1
        self.robot = utils.VrepCommunication()
        self.termination_height = 0.13

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork().to(self.device)
        # self.policy_net.load_state_dict(torch.load(r"/home/george/Desktop/" \
        #     r"Github/Reinforcement_Learning/sac.pt"))
        print(sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad))
        # print(sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad))

        self.soft_q_net1 = SoftQNetwork().to(self.device)
        print(sum(p.numel() for p in self.soft_q_net1.parameters() if p.requires_grad))

        self.soft_q_net2 = SoftQNetwork().to(self.device)

        self.target_soft_q_net1 = SoftQNetwork().to(self.device)

        self.target_soft_q_net2 = SoftQNetwork().to(self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), \
            self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), \
            self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.target_soft_q_net1.train()
        self.target_soft_q_net2.train()

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.learning_rate = 1e-5
        self.gamma = 0.99
        self.soft_tau = 1e-2

        self.target_entropy = -np.prod(5).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.learning_rate)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        replay_buffer_size = 500
        self.replay_buffer = buffer.ReplayMemory(replay_buffer_size, self.batch_size)
        self.action_x_lower_bounds = -0.05
        self.action_x_upper_bounds = 0.05
        self.action_y_lower_bounds = -0.05
        self.action_y_upper_bounds = 0.05
        self.action_z_lower_bounds = -0.05
        self.action_z_upper_bounds = 0.05
        self.action_theta_upper_bound = 1.5533430343
        self.action_theta_lower_bound = 0.01745329252
        self.hand_status = None

    def start_vrep(self, again=False):
        if again:
            self.robot.end_communication()
            self.robot.establish_communication()
            self.robot.start_vrep()
            self.robot.initialise()
            self.robot.pick_color()
            self.robot.add_object()
            self.robot.reset_object_state()
            self.robot.get_initial_position()
        else:
            self.robot.establish_communication()
            self.robot.start_vrep()
            self.robot.initialise()
            self.robot.pick_color()
            self.robot.add_object()
            self.robot.reset_object_state()
            self.robot.get_initial_position()

    def main_trainer(self):
        rewards = []
        successes = []
        self.start_vrep()
        k = 0
        for j in range(1, 10000000):

            self.robot.set_initial_position()
            image = self.robot.get_image()
            episode_reward = 0
            height = 0.12765
            self.hand_status = 1
            numerical_state = np.array([self.hand_status, height]) #initially for every episode.
            for step in range(30): #the allowed number of steps per episode is 30.
                action = self.select_action(image, numerical_state) #takes the action
                reward, done, success, next_image, next_numerical_state, label = \
                    self.execute_action(action)
                if not label:

                    if done:
                        next_image = None
                        next_numerical_state = None

                    self.replay_buffer.push(image, numerical_state, action, self.reward_scale* \
                    reward, next_image, next_numerical_state, done)
                    self.replay_buffer.store_at_disk()
                    self.replay_buffer.empty()
                    k += 1

                    image = next_image
                    numerical_state = next_numerical_state
                    episode_reward += reward

                    if k > self.batch_size:
                        self.update()

                    if done:
                        break

                else: #if trouble opening the hand then
                    if k > self.batch_size:
                        self.update()
                    break

            rewards.append(episode_reward)
            successes.append(success)

            if j%50 == 0:
                self.plot_rewards(rewards)
                self.plot_success_rate(successes)

            if j%500 == 0:
                self.start_vrep(True)

            if j%100 == 0:
                torch.save(self.policy_net.state_dict(), r"/home/george/Desktop/" \
                    r"Github/Reinforcement_Learning/sac.pt")

            # self.robot.delete_texture()
            # self.robot.domain_randomize()
            #
            # if j%100==0:
            #     self.robot.delete_object()
            #     self.robot.delete_texture()
            #     self.robot.add_object()
            #     self.robot.reset_object_state()
            #     continue

            self.robot.reset_object_state()

    def plot_rewards(self, rewards):
        steps = [i*50 for i in range(len(rewards)/50)]
        avg = []
        for i in range(len(rewards)/50):
            reward_list = rewards[i*50:(i+1)*50]
            avg.append(sum(reward_list)/float(len(reward_list)))
        plt.plot(steps, avg)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.suptitle('mean reward of last 50 episodes')
        plt.show()
        plt.pause(3)
        plt.savefig('/home/george/Desktop/Github/Reinforcement_Learning/reward_plot.jpg')
        plt.cla()
        plt.close()

    def plot_success_rate(self, success):
        steps = [i*50 for i in range(len(success)/50)]
        rate = []
        for i in range(len(success)/50):
            reward_list = success[i*50:(i+1)*50]
            rate.append(reward_list.count(1))

        plt.plot(steps, rate)
        plt.xlabel('episodes')
        plt.ylabel('successful attempts')
        plt.suptitle('success rate of last 50 episodes')
        plt.show()
        plt.pause(3)
        plt.savefig('/home/george/Desktop/Github/Reinforcement_Learning/success_plot.jpg')
        plt.cla()
        plt.close()

    def update(self):
        self.policy_net.train()
        self.soft_q_net1.train()
        self.soft_q_net2.train()

        image, numerical_state, action, reward, non_final_mask, non_final_next_image, \
            non_final_next_numerical_state, done = self.replay_buffer.sample()
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)

        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(image, \
            numerical_state)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        predicted_new_q_value = torch.min(self.soft_q_net1(image, numerical_state, \
            new_action.to(self.device)), self.soft_q_net2(image, numerical_state, new_action. \
            to(self.device)))
        policy_loss = (alpha*log_prob - predicted_new_q_value).mean()

        predicted_q_value1 = self.soft_q_net1(image, numerical_state, action)
        predicted_q_value2 = self.soft_q_net2(image, numerical_state, action)
        new_next_action, new_log_prob, new_epsilon, new_mean, new_log_std = self.policy_net. \
            evaluate(non_final_next_image, non_final_next_numerical_state)
        predicted_new_next_q_value = torch.zeros(self.batch_size, device=self.device).unsqueeze(1)

        predicted_new_next_q_value[non_final_mask] = torch.min(self.target_soft_q_net1 \
            (non_final_next_image, non_final_next_numerical_state, new_next_action.to(self. \
            device)), self.target_soft_q_net2(non_final_next_image, \
            non_final_next_numerical_state, new_next_action.to(self.device))) - alpha*new_log_prob

        predicted_new_next_q_value = predicted_new_next_q_value.to(self.device)

        target_q_value = reward + (1-done) * self.gamma * predicted_new_next_q_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        print('q value loss 1: ', q_value_loss1)
        print('q value loss 2: ', q_value_loss2)
        print('policy loss: ', policy_loss)
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), 1)
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), 1)
        self.soft_q_optimizer2.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_soft_q_net1.parameters(), \
            self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data \
                * self.soft_tau)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), \
            self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data \
                * self.soft_tau)

        self.policy_net.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()

    global NUM_ACTIONS#, flag, counter
    NUM_ACTIONS = 0
    # flag = True
    # counter = 0
    def select_action(self, image_state, numerical_state):
        global NUM_ACTIONS#, flag, counter
        # if NUM_ACTIONS % 30000 == 0:
        #     flag = True
        #     counter = 0
        if NUM_ACTIONS < 1000:
            action_x = np.random.uniform(low=-0.05, high=0.05, size=1)[0]
            action_y = np.random.uniform(low=-0.05, high=0.05, size=1)[0]
            action_z = np.random.uniform(low=-0.05, high=0.05, size=1)[0]
            action_close = np.random.uniform()
            action = [action_x, action_y, action_z, action_close]
            # counter += 1
            # if counter == 1000:
            #     flag = False
        else:
            img_tensor = torch.from_numpy(image_state)
            numerical_state_tensor = torch.from_numpy(numerical_state)
            action = self.policy_net.get_action(img_tensor.unsqueeze(0).permute(0, 3, 1, 2).float()\
                .to(self.device), numerical_state_tensor.unsqueeze(0).float().to(self.device))

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
        # if action[3] < self.action_theta_lower_bound:
        #     action[3] = self.action_theta_lower_bound
        # elif action[3] > self.action_theta_upper_bound:
        #     action[3] = self.action_theta_upper_bound
        if action[3] < 0:
            action[3] = 0
        elif action[3] >= 0:
            action[3] = 1
        NUM_ACTIONS += 1
        return action

    def execute_action(self, action):
        label = False
        success, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
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
            vrep.simxSetObjectPosition(self.robot.client_id, self.robot.sawyer_target_handle, \
                -1, (sawyer_target_position[0] + move_step[0], sawyer_target_position[1] \
                +move_step[1], sawyer_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
            _, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.client_id, \
                self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.robot.client_id)
            vrep.simxGetPingTime(self.robot.client_id)
        vrep.simxSetObjectPosition(self.robot.client_id, self.robot.sawyer_target_handle, -1, \
            (sawyer_target_position[0] + remaining_distance[0], sawyer_target_position[1] + \
            remaining_distance[1], sawyer_target_position[2] + remaining_distance[2]), \
            vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.robot.client_id)
        vrep.simxGetPingTime(self.robot.client_id)

        _, sawyer_orientation = vrep.simxGetObjectOrientation(self.robot.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if(action[3] - sawyer_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((action[3]-sawyer_orientation[1])/rotation_step))
        for step_iter in range(num_rotation_steps):
            vrep.simxSetObjectOrientation(self.robot.client_id, self.robot. \
                sawyer_target_handle, -1, (sawyer_orientation[0], sawyer_orientation[1] + \
                rotation_step, sawyer_orientation[2]), vrep.simx_opmode_blocking)
            _, sawyer_orientation = vrep.simxGetObjectOrientation(self.robot.client_id, \
                self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.robot.client_id)
            vrep.simxGetPingTime(self.robot.client_id)

        vrep.simxSetObjectOrientation(self.robot.client_id, self.robot.sawyer_target_handle, -1, \
            (sawyer_orientation[0], action[3], sawyer_orientation[2]), vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.robot.client_id)
        vrep.simxGetPingTime(self.robot.client_id)

        if action[3] == 1: #open hand
            since = time.time()
            _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.motor_handle,\
                vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.robot.client_id, self.robot.motor_handle, 20,\
                vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motor_handle, -0.5,\
                vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.robot.client_id)
            vrep.simxGetPingTime(self.robot.client_id)
            while dist > -1e-06:
                _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.\
                    motor_handle, vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motor_handle, \
                    -0.5, vrep.simx_opmode_blocking)
                vrep.simxSynchronousTrigger(self.robot.client_id)
                vrep.simxGetPingTime(self.robot.client_id)
                if time.time()-since > 20:
                    label = True
                    print('trouble opening the gripper')
                    break
            vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motor_handle, 0.0,\
                vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.robot.client_id)
            vrep.simxGetPingTime(self.robot.client_id)
            self.hand_status = 1

        elif action[3] == 0:
            since = time.time()
            _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.motor_handle,\
                vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.robot.client_id, self.robot.motor_handle, 100,\
                vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motor_handle, 0.5,\
                vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.robot.client_id)
            vrep.simxGetPingTime(self.robot.client_id)
            while dist < 0.0345:
                _, dist = vrep.simxGetJointPosition(self.robot.client_id, self.robot.\
                    motor_handle, vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motor_handle, 0.5,\
                    vrep.simx_opmode_blocking)
                vrep.simxSynchronousTrigger(self.robot.client_id)
                vrep.simxGetPingTime(self.robot.client_id)
                if time.time()-since > 5:
                    if self.robot.has_object():
                        break
                    print(dist)
                    print('trouble closing the gripper')
            vrep.simxSetJointTargetVelocity(self.robot.client_id, self.robot.motor_handle, 0.5,\
                vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.robot.client_id)
            vrep.simxGetPingTime(self.robot.client_id)
            self.hand_status = 0

        if not label:
            image = self.robot.get_image()
            success, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.client_id,\
                self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            numerical_state = np.array([self.hand_status, sawyer_target_position[2]])
            new_sawyer_position = sawyer_target_position
            success, object_position = vrep.simxGetObjectPosition(self.robot.client_id,\
                self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
            reward, done, success = self.reward(action, old_sawyer_position, new_sawyer_position,\
                object_position)
        else:
            done = None
            reward = None
            success = None
            image = None
            numerical_state = None
        return reward, done, success, image, numerical_state, label

    def reward(self, action, old_sawyer_position, new_sawyer_position, object_position):
        done = False
        reward = -0.1
        success = 0

        if self.hand_status == 1 and action[3] == 1: #when moving towards the object
            before = np.array(old_sawyer_position) - np.array(object_position)
            after = np.array(new_sawyer_position) - np.array(object_position)
            old_distance = np.linalg.norm(before) #5
            new_distance = np.linalg.norm(after) #3
            reward += old_distance-new_distance #incentivize to move closer to the object

        elif self.hand_status == 0:
            distance = np.linalg.norm(np.array(object_position)-np.array(new_sawyer_position))
            hold = self.robot.has_object()
            if hold: #holds the object
                self.robot.lift_arm()
                label = self.robot.successful_grasp()
                if label == 0:
                    done = True
                    reward += -distance
                elif label == 1:
                    reward += 10
                    done = True
                    success = 1
            else:
                reward += -distance

        if not self.robot.check_object_inbounds() or new_sawyer_position[2] > \
            self.termination_height:
            done = True

        return reward, done, success

class SoftQNetwork(nn.Module):

    def __init__(self, num_classes=32):
        super(SoftQNetwork, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) #320
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) #16
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) #8
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) #4
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1) #2
        self.dropout = nn.Dropout(0.5)
        self.fc_model_0 = nn.Linear(in_features=2*2*64, out_features=256)
        self.fc_model_1 = nn.Linear(in_features=256, out_features=128)
        self.fc_model_2 = nn.Linear(in_features=128, out_features=64)
        self.fc_model_3 = nn.Linear(in_features=64, out_features=32)

        self.fc0_1 = nn.Linear(2, 16)
        self.fc0_2 = nn.Linear(16, 32)

        self.fc1 = nn.Linear(num_classes, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        self.fc1_1 = nn.Linear(4, 16)
        self.fc1_2 = nn.Linear(16, 32)

        # self.fc0_4 = nn.Linear(128, 128)

        init_w = 1e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, data, numerical_state, action):
        data = self.bn0(self.relu(self.conv0(data)))
        data = self.bn1(self.relu(self.conv1(data)))
        data = self.bn2(self.relu(self.conv2(data)))
        data = self.bn3(self.relu(self.conv3(data)))
        data = self.relu(self.conv4(data))
        (_, pixels, height, width) = data.data.size()
        data = data.view(-1, pixels*height*width)
        data = self.relu(self.fc_model_0(data))
        data = self.relu(self.fc_model_1(data))
        data = self.relu(self.fc_model_2(data))
        data = self.relu(self.fc_model_3(data))
        numerical_state = self.relu(self.fc0_1(numerical_state))
        numerical_state = self.relu(self.fc0_2(numerical_state))

        action = self.relu(self.fc1_1(action))
        action = self.relu(self.fc1_2(action))

        new_vector = data + numerical_state + action
        new_vector = self.relu(self.fc1(new_vector))
        new_vector = self.relu(self.fc2(new_vector))
        output = self.fc3(new_vector)
        return output

class PolicyNetwork(nn.Module):
    def __init__(self, num_classes=32):
        super(PolicyNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc_model_0 = nn.Linear(in_features=2*2*64, out_features=256)
        self.fc_model_1 = nn.Linear(in_features=256, out_features=128)
        self.fc_model_2 = nn.Linear(in_features=128, out_features=64)
        self.fc_model_3 = nn.Linear(in_features=64, out_features=32)

        self.fc0_1 = nn.Linear(2, 16)
        self.fc0_2 = nn.Linear(16, 32)

        self.fc1 = nn.Linear(num_classes, 64)
        self.fc2 = nn.Linear(64, 64)

        self.mean_linear = nn.Linear(64, 4)

        init_w = 1e-3
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(64, 4)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_min = -20
        self.log_std_max = -3

    def forward(self, data, numerical_state):
        data = self.bn0(self.relu(self.conv0(data)))
        data = self.bn1(self.relu(self.conv1(data)))
        data = self.bn2(self.relu(self.conv2(data)))
        data = self.bn3(self.relu(self.conv3(data)))
        data = self.relu(self.conv4(data))
        (_, pixels, height, width) = data.data.size()
        data = data.view(-1, pixels*height*width)
        data = self.relu(self.fc_model_0(data))
        data = self.relu(self.fc_model_1(data))
        data = self.relu(self.fc_model_2(data))
        data = self.relu(self.fc_model_3(data))
        numerical_state = self.relu(self.fc0_1(numerical_state))
        numerical_state = self.relu(self.fc0_2(numerical_state))
        new_vector = data + numerical_state #Nx128

        new_vector = self.relu(self.fc1(new_vector))
        new_vector = self.relu(self.fc2(new_vector))

        mean = self.mean_linear(new_vector)
        log_std = self.log_std_linear(new_vector)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, data, numerical_state, epsilon=1e-6): #must check how to fine-tune it.
        mean, log_std = self.forward(data, numerical_state)
        std = torch.exp(log_std)
        policy = (mean + std * Normal(torch.zeros(4), torch.ones(4)).sample().to(self.device))
        policy.requires_grad_()
        action = torch.tanh(policy)
        log_prob = Normal(torch.zeros(4).to(self.device), torch.ones(4).to(self.device)).\
            log_prob(policy) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, policy, mean, log_std

    def get_action(self, data, numerical_state):
        mean, log_std = self.forward(data, numerical_state)
        std = torch.exp(log_std)
        normal = Normal(torch.zeros(4), torch.ones(4))
        policy = normal.sample().to(self.device)
        action = torch.tanh(mean + std*policy)
        action = action.detach().cpu().numpy()
        return action[0]

SAC = Main()
SAC.main_trainer()
