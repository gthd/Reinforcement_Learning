from PIL import Image  as I
import signal
import cv2
import os
import numpy as np
import vrep
from subprocess import Popen
import time
import gzip
import shutil
import array
np.random.seed(24)
class Vrep_Communication:

    def __init__(self, client_id=None, process='', port=19997,host='127.0.0.1'):

        self.states = []
        self.object_position = []
        self.object_orientation = []
        self.workspace_limits = np.asarray([[1.028, 1.242], [1.1, 1.278], [-0.0001, 0.4]])
        self.object_handle = []

        # Read files in object mesh directory
        self.obj_mesh_texture_dir = "/homes/gt4118/Desktop/Robot_Learning/textures/"

        self.texture_list = os.listdir(self.obj_mesh_texture_dir)

        self.client_id = client_id
        self.process = process
        self.port_num = port
        self.host = host
        self.vrep_path = "/homes/gt4118/Desktop/V-REP_PRO_EDU_V3_6_1_Ubuntu18_04/vrep.sh"
        vrep.simxFinish(-1) # just in case a previous connection is open then close it

    def pick_color(self):
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                       [89.0, 161.0, 79.0], # green
                                       [156, 117, 95], # brown
                                       [242, 142, 43], # orange
                                       [237.0, 201.0, 72.0], # yellow
                                       [186.0, 176, 172], # gray
                                       [255.0, 87.0, 89.0], # red
                                       [176, 122, 161], # purple
                                       [118, 183, 178], # cyan
                                       [255, 157, 167],
                                       [0, 0, 128],
                                       [128, 0, 0],
                                       [170, 110, 40],
                                       [128, 128, 0],
                                       [170, 255, 195],
                                       [70, 240, 240],
                                       [250, 190, 190],
                                       [210, 245, 60],
                                       [240, 50, 230],
                                       [255, 215, 180],
                                       [0, 128, 128],
                                       [60, 180, 75]])/255.0
        num = np.random.randint(low=0,high=9,size=1)[0]
        self.color = self.color_space[num]

    def end_communication(self):
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        vrep.simxFinish(self.client_id)
        pgrp = os.getpgid(self.process.pid)
        os.killpg(pgrp, signal.SIGKILL)

    def establish_communication(self):
        remote_api_string = '-h -gREMOTEAPISERVERSERVICE_' + str(self.port_num) + '_FALSE_TRUE'
        parent_dir = os.path.abspath(os.path.join("..", os.pardir))
        args = [self.vrep_path, remote_api_string]
        self.process = Popen(args, preexec_fn=os.setsid)
        time.sleep(6)

    def start_vrep(self):
        self.client_id = vrep.simxStart(self.host, self.port_num, True, True, 5000, 5)
        return_code = vrep.simxSynchronous(self.client_id, enable=True)
        self.check_for_errors(return_code)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        scene_path = dir_path + '/my_scene3.ttt'
        return_code = vrep.simxLoadScene(self.client_id, scene_path, 0, vrep.simx_opmode_blocking)
        self.check_for_errors(return_code)
        return_code = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
        self.check_for_errors(return_code)

    def check_for_errors(self, code):
        if code == vrep.simx_return_ok:
            return
        elif code == vrep.simx_return_novalue_flag:
            pass
        elif code == vrep.simx_return_timeout_flag:
            raise RuntimeError('The function timed out (probably the network is down or too slow)')
        elif code == vrep.simx_return_illegal_opmode_flag:
            raise RuntimeError('The specified operation mode is not supported for the given function')
        elif code == vrep.simx_return_remote_error_flag:
            raise RuntimeError('The function caused an error on the server side (e.g. an invalid handle was specified)')
        elif code == vrep.simx_return_split_progress_flag:
            raise RuntimeError('The communication thread is still processing previous split command of the same type')
        elif code == vrep.simx_return_local_error_flag:
            raise RuntimeError('The function caused an error on the client side')
        elif code == vrep.simx_return_initialize_error_flag:
            raise RuntimeError('A connection to vrep has not been made yet. Have you called connect()? (Port num = ' + str(return_code.port_num))

    def call_lua_function(self, lua_function, ints=[], floats=[], strings=[], bytes=bytearray(), opmode=vrep.simx_opmode_blocking):
        return_code, out_ints, out_floats, out_strings, out_buffer = vrep.simxCallScriptFunction(self.client_id, 'remote_api', vrep.sim_scripttype_customizationscript, lua_function, ints, floats, strings, bytes, opmode)
        self.check_for_errors(return_code)
        return out_ints, out_floats, out_strings, out_buffer

    def initialise(self):
        return_code, self.robot_handle = vrep.simxGetObjectHandle(self.client_id, 'Sawyer', vrep.simx_opmode_blocking)
        sim_ret, self.Sawyer_target_handle = vrep.simxGetObjectHandle(self.client_id,'Sawyer_Target',vrep.simx_opmode_blocking)
        returnCode, self.motorHandle = vrep.simxGetObjectHandle(self.client_id, 'BaxterGripper_closeJoint', vrep.simx_opmode_blocking)
        returnCode, self.proxSensorHandle = vrep.simxGetObjectHandle(self.client_id, 'BaxterGripper_attachProxSensor', vrep.simx_opmode_blocking)
        self.joint_handles = []
        self.indicator_handles = []
        for i in range(7):
            returnCode, joint_handle = vrep.simxGetObjectHandle(self.client_id, 'Sawyer_joint' + str(i+1), vrep.simx_opmode_blocking)
            self.joint_handles.append(joint_handle)
        return_code, self.vision_handle = vrep.simxGetObjectHandle(self.client_id, 'Vision_sensor', vrep.simx_opmode_blocking)
        for i in range(15):
            returnCode, indicator_handle = vrep.simxGetObjectHandle(self.client_id, 'Plane' + str(i+3), vrep.simx_opmode_blocking)
            self.indicator_handles.append(indicator_handle)

    def add_object(self):
        #self.ind = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        self.ind = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        size = np.random.uniform(low=0.02, high=0.03, size=3)
        random = np.random.uniform(low=0, high=1, size=1)
        if random < 0.33:
            type=0
        elif random < 0.66:
            type = 1
        elif random <= 1:
            type = 2
        ret_resp, self.object_handle,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'addObject', [0], [0.01, 0.01, size[0], size[1], size[2], self.color[0],self.color[1],self.color[2]] , ['Shape_'+str(self.ind), self.obj_mesh_texture_dir + self.texture_list[self.ind]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_mat(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_mat', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_floor(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_floor', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_pads(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=2)
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_pads', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb[0]], self.obj_mesh_texture_dir + self.texture_list[numb[1]]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_gripper(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_gripper', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_link7(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_link7', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_link6(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_link6', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_link5(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_link5', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_link1(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_link1', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_link0(self):
        numb = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_link0', [], [] , [self.obj_mesh_texture_dir + self.texture_list[numb]], bytearray(),vrep.simx_opmode_blocking)

    def randomize_links(self):
        self.randomize_link5()
        self.randomize_link0()
        self.randomize_link6()
        self.randomize_link7()
        self.randomize_link1()

    def randomize_camera(self):
        self.vision_handle
        _, pos = vrep.simxGetObjectPosition(self.client_id, self.vision_handle, -1, vrep.simx_opmode_blocking)
        y_pos = np.random.uniform(low=1.4348, high=1.4948, size=1)[0]
        z_pos = np.random.uniform(low=0.0924, high= 0.2424, size=1)[0]
        x_pos = pos[0]
        position = np.array([x_pos, y_pos, z_pos])
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_camera', [], position , [], bytearray(),vrep.simx_opmode_blocking)

    def randomize_light(self):
        probs = np.random.uniform(low=0, high=1, size=4)
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'randomize_light', [], probs, [], bytearray(),vrep.simx_opmode_blocking)

    def domain_randomize(self):
        self.randomize_light()
        self.randomize_camera()
        self.randomize_links()
        self.randomize_gripper()
        self.randomize_pads()
        self.randomize_floor()
        self.randomize_mat()

    def show_robot(self, show=True):
        if show:
            ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'show_robot', [1], [] , [], bytearray(),vrep.simx_opmode_blocking)
        else:
            ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'show_robot', [0], [] , [], bytearray(),vrep.simx_opmode_blocking)

    def get_image(self):
        for i in range(4):
            err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.vision_handle, 0, vrep.simx_opmode_blocking)
            self.check_for_errors(err)
        image_byte_array = array.array('b', image)
        image_buffer = I.frombuffer("RGB", (resolution[0],resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
        img2 = np.asarray(image_buffer)
        img2_0 = img2[:,:,0]
        img2_1 = img2[:,:,1]
        img2_2 = img2[:,:,2]
        img2 = np.dstack((img2_2,img2_1, img2_0))
        #img2 = img2[::-1,:,:]
        return img2

    def reset_object_position_and_orientation(self):
        self.object_position = []
        self.object_orientation = []
        for obj in self.object_handle:
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0]) * np.random.random_sample() + self.workspace_limits[0][0]
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0]) * np.random.random_sample() + self.workspace_limits[1][0]
            self.object_position = [drop_x, drop_y, 1.0208e-02]
            self.object_orientation = [-1.571, 2*np.pi*np.random.random_sample(), -1.571]
            vrep.simxSetObjectPosition(self.client_id, obj, -1, self.object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.client_id, obj, self.robot_handle, self.object_orientation, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

    def set_back(self):
        vrep.simxSetObjectPosition(self.client_id, self.object_handle[0], -1, self.object_position, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.client_id, self.object_handle[0], self.robot_handle, self.object_orientation, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def open_hand(self):#this should work to place gripper at initial open position
        _, dist = vrep.simxGetJointPosition(self.client_id, self.motorHandle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.client_id, self.motorHandle, 20, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.client_id, self.motorHandle, -0.5, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)
        start_time = time.time()
        while dist > -1e-06:#8.22427227831e-06:#-1.67e-06: # Block until gripper is fully open
            sim_ret, dist = vrep.simxGetJointPosition(self.client_id, self.motorHandle, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.client_id, self.motorHandle, -0.5, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)
            end_time = time.time()
            # if (end_time-start_time > 2):
            #     self.restart()
        vrep.simxSetJointTargetVelocity(self.client_id, self.motorHandle, 0.0, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def close_hand(self):
        vrep.simxSetJointForce(self.client_id, self.motorHandle, 100, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.client_id, self.motorHandle, 0.5, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def lift_arm(self):
        success, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        lift_position_x = 1.137
        lift_position_y = 1.2151
        lift_position_z = 0.18
        self.lift_position = np.array([lift_position_x, lift_position_y, lift_position_z])
        move_direction = np.asarray([self.lift_position[0] - Sawyer_target_position[0], self.lift_position[1] - Sawyer_target_position[1], self.lift_position[2] - Sawyer_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.01*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.01))
        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.client_id,self.Sawyer_target_handle,-1,(Sawyer_target_position[0] + move_step[0], Sawyer_target_position[1] + move_step[1], Sawyer_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            sim_ret, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id,self.Sawyer_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)
        success = vrep.simxSetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, self.lift_position, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def shake_arm(self):
        self.move_to_positions = []
        for i in range(4):
            self.move_to_positions.append([np.random.uniform(low=1.018, high=1.223,size=1)[0], np.random.uniform(low=0.951, high=1.294,size=1)[0], np.random.uniform(low=0.1, high=0.18,size=1)[0]])

        for i in range(len(self.move_to_positions)):
            success, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            move_position = self.move_to_positions[i]
            move_direction = np.asarray([move_position[0] - Sawyer_target_position[0], move_position[1] - Sawyer_target_position[1], move_position[2] - Sawyer_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.01*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.01))
            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.client_id,self.Sawyer_target_handle,-1,(Sawyer_target_position[0] + move_step[0], Sawyer_target_position[1] + move_step[1], Sawyer_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id,self.Sawyer_target_handle,-1,vrep.simx_opmode_blocking)
                vrep.simxSynchronousTrigger(self.client_id)
                vrep.simxGetPingTime(self.client_id)
            success = vrep.simxSetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, move_position, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

    def successful_grasp(self):
        i = 0
        while True:
            i += 1
            retCode, pos = vrep.simxGetObjectPosition(self.client_id, self.object_handle[0], -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)
            if pos[2] < 0.02:
                self.success = False
                return 0
            if i > 5:
                self.success = True
                return 1

    def get_initial_position(self):
        retCode, self.init_endpoint_pos = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        retCode, self.init_endpoint_ori = vrep.simxGetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        self.joint_positions = []
        self.joint_orientation = []
        for joint in self.joint_handles:
            success, pos = vrep.simxGetJointPosition(self.client_id, joint, vrep.simx_opmode_blocking)
            success, orientation = vrep.simxGetObjectOrientation(self.client_id, joint, -1, vrep.simx_opmode_blocking)
            self.joint_positions.append(pos)
            self.joint_orientation.append(orientation)

    def set_initial_position(self):
        self.open_hand()
        success, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        move_direction = np.asarray([self.init_endpoint_pos[0] - Sawyer_target_position[0], self.init_endpoint_pos[1] - Sawyer_target_position[1], self.init_endpoint_pos[2] - Sawyer_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.01*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.01))
        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.client_id,self.Sawyer_target_handle,-1,(Sawyer_target_position[0] + move_step[0], Sawyer_target_position[1] + move_step[1], Sawyer_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            sim_ret, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id,self.Sawyer_target_handle, -1,vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)
        vrep.simxSetObjectPosition(self.client_id,self.Sawyer_target_handle, -1,self.init_endpoint_pos,vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)
        vrep.simxSetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, (self.init_endpoint_ori[0], self.init_endpoint_ori[1], self.init_endpoint_ori[2]), vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

        for i in range(len(self.joint_handles)): #maybe the problem comes from setting directly the position so I should set it more gradually
            vrep.simxSetJointPosition(self.client_id, self.joint_handles[i], self.joint_positions[i], vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.client_id, self.joint_handles[i], -1, self.joint_orientation[i], vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)


    def delete_object(self):
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'delete_shape', self.object_handle, [], [], bytearray(), vrep.simx_opmode_blocking)

    def add_object_from_list(self): #have to set the object on the scene first
        ind = np.random.randint(low=0,high=len(self.texture_list),size=1)[0]
        size = np.random.uniform(low=0.02, high=0.04, size=3)
        mass = np.random.uniform(low=0.1, high=1.5, size=1)
        damping = np.random.uniform(low=0.01, high=0.2,size=2)
        ret_resp, self.object_handle,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'addObject2', [], [damping[0], damping[1], size[0], size[1], size[2], mass[0]] , ['Shape_'+str(ind), self.obj_mesh_texture_dir + self.texture_list[ind], self.obj_mesh_dir + self.mesh_list[ind]], bytearray(),vrep.simx_opmode_blocking)

    def delete_texture(self):
        ret_resp, _,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'retrieve_texture_planes', [], [] , [], bytearray(),vrep.simx_opmode_blocking)

    def randomize_gripper_location(self):
        _, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        _, Sawyer_target_orientation = vrep.simxGetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)

        x_pos = np.random.uniform(low=1.028, high=1.242, size=1)[0]
        y_pos = np.random.uniform(low=1.1, high=1.278, size=1)[0]
        new_position = np.array([x_pos, y_pos, Sawyer_target_position[2]])
        orientation = np.random.uniform(low=0.01745329252, high=1.5533430343, size=1)[0]
        new_orientation = np.array([Sawyer_target_orientation[0], orientation, Sawyer_target_orientation[2]])

        success, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        move_direction = np.asarray([new_position[0] - Sawyer_target_position[0], new_position[1] - Sawyer_target_position[1], new_position[2] - Sawyer_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.01*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.01))
        remaining_magnitude = -num_move_steps * 0.01 + move_magnitude
        remaining_distance = remaining_magnitude * move_direction/move_magnitude

        for step_iter in range(num_move_steps): #selects action and executes action
            vrep.simxSetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, (Sawyer_target_position[0] + move_step[0], Sawyer_target_position[1] + move_step[1], Sawyer_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
            sim_ret, Sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

        vrep.simxSetObjectPosition(self.client_id, self.Sawyer_target_handle, -1, (Sawyer_target_position[0] + remaining_distance[0], Sawyer_target_position[1] + remaining_distance[1], Sawyer_target_position[2]+ remaining_distance[2]),vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

        sim_ret, sawyer_orientation = vrep.simxGetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (new_orientation[1] - sawyer_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((new_orientation[1] - sawyer_orientation[1])/rotation_step))

        for step_iter in range(num_rotation_steps):
            vrep.simxSetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, (sawyer_orientation[0], sawyer_orientation[1] + rotation_step, sawyer_orientation[2]), vrep.simx_opmode_blocking)
            sim_ret, sawyer_orientation = vrep.simxGetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

        vrep.simxSetObjectOrientation(self.client_id, self.Sawyer_target_handle, -1, (sawyer_orientation[0], new_orientation[1], sawyer_orientation[2]), vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def check_collision(self):
        results = []
        for i in range(len(self.indicator_handles)):
            ret_resp, result,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript,'check_collision', [self.object_handle[0], self.indicator_handles[i]], [] , [], bytearray(),vrep.simx_opmode_blocking)
            results.append(result[0])
        if 1 in results:
            return True
        return False

    def check_detection(self):
        ret_resp, result,_,_,_ = vrep.simxCallScriptFunction(self.client_id, 'remote_api',vrep.sim_scripttype_childscript, 'check_proximity', [self.object_handle[0], self.proxSensorHandle], [] , [], bytearray(),vrep.simx_opmode_blocking)

        if result[0] == 1:
            return True
        return False

    def has_object(self):
        #label1 = self.check_collision()
        label2 = self.check_detection()
        return label2

    def check_object_inbounds(self):
        retCode, obj_pos = vrep.simxGetObjectPosition(self.client_id, self.object_handle[0], -1, vrep.simx_opmode_blocking)
        if obj_pos[0] < 1.028 or obj_pos[0] > 1.242 or obj_pos[1] < 1.1 or obj_pos[1] > 1.278:
            return False
        return True
