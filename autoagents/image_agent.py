import os
import math
import time
import json
import yaml
import lmdb
import numpy as np
from PIL import Image
import torch
import wandb
import carla
import random
import string

from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs

from rails.models import EgoModel, CameraModel
from autoagents.waypointer import Waypointer

# jxy: addition; (add display.py and fix RoutePlanner.py or Waypointer.py)
from team_code.display import HAS_DISPLAY, Saver, debug_display
# addition from team_code/map_agent.py
from carla_project.src.common import CONVERTER, COLOR
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights


def get_entry_point():
    return 'ImageAgent'

class ImageAgent(AutonomousAgent):
    
    """
    Trained image agent
    """
    
    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.SENSORS
        self.num_frames = 0
        self.config_path = path_to_conf_file
        self.wall_start = time.time()
        self.initialized = False

        return AgentSaver

        # jxy: add return AgentSaver and init_ads (setup keep 5 lines); rm save_path;
    def init_ads(self, path_to_conf_file):

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')

        self.image_model = CameraModel(config).to(self.device)
        self.image_model.load_state_dict(torch.load(self.main_model_dir))
        self.image_model.eval()

        self.vizs = []

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_evaluate_WorldOnRails')
            
        self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None

    def destroy(self): # jxy mv before _init
        if len(self.vizs) == 0:
            return

        self.flush_data()

        del self.waypointer
        del self.image_model
        torch.cuda.empty_cache()
        super().destroy()

    def _init(self):

        self.initialized = True

        # del self.net # jxy from destroy to here, as twice destroy in a round
        super()._init() # jxy add
    
    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })
            
        self.vizs.clear()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.stitch_camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB'},
            # jxy: addition from team_code/map_agent.py
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 0.0, 'y': 0.0, 'z': 100.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 512, 'height': 512, 'fov': 5 * 10.0,
                'id': 'map'
                },
        ]
        
        return sensors

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        
        _, wide_rgb = input_data.get(f'Wide_RGB')
        _, narr_rgb = input_data.get(f'Narrow_RGB')

        # Crop images
        _wide_rgb = wide_rgb[self.wide_crop_top:,:,:3]
        _narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,:3]

        _wide_rgb = _wide_rgb[...,::-1].copy()
        _narr_rgb = _narr_rgb[...,::-1].copy()

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')


        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        # jxy: add pos
        (_, _, cmd), pos = self.waypointer.tick(gps)

        spd = ego.get('speed')
        
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3

        _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0,3,1,2).to(self.device)
        
        if self.all_speeds:
            steer_logits, throt_logits, brake_logits = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value)
            # Interpolate logits
            steer_logit = self._lerp(steer_logits, spd)
            throt_logit = self._lerp(throt_logits, spd)
            brake_logit = self._lerp(brake_logits, spd)
        else:
            steer_logit, throt_logit, brake_logit = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value, spd=torch.tensor([spd]).float().to(self.device))

        
        action_prob = self.action_prob(steer_logit, throt_logit, brake_logit)

        brake_prob = float(action_prob[-1])

        steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
        throt = float(self.throts @ torch.softmax(throt_logit, dim=0))

        steer, throt, brake = self.post_process(steer, throt, brake_prob, spd, cmd_value)

        
        rgb = np.concatenate([wide_rgb, narr_rgb[...,:3]], axis=1)
        
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), spd, cmd=cmd_value+1))

        if len(self.vizs) > 1000:
            self.flush_data()

        self.num_frames += 1

        control = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

        # jxy addition:
        self.step = self.num_frames
        tick_data = {
                'rgb': rgb,
                'gps': pos,
                'speed': spd,
                'compass': -1,
                }
        tick_data['far_command'] = cmd
        # tick_data['R_pos_from_head'] = R
        tick_data['offset_pos'] = np.array([pos[0], pos[1]])
        # from team_code/map_agent.py:
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        tick_data['topdown'] = COLOR[CONVERTER[topdown]]

        if HAS_DISPLAY: # jxy: change
            debug_display(tick_data, control.steer, control.throttle, control.brake, self.step)

        self.record_step(tick_data, control, ) # jxy: add
        return control

    # jxy: add record_step
    def record_step(self, tick_data, control, pred_waypoint=[]):
        # draw pred_waypoint
        # if len(pred_waypoint):
            # pred_waypoint[:,1] *= -1
            # pred_waypoint = tick_data['R_pos_from_head'].dot(pred_waypoint.T).T
        self.waypointer.run_step2(pred_waypoint, is_gps=False, store=False) # metadata['wp_1'] relative to ego head (as y)
        # addition: from leaderboard/team_code/auto_pilot.py
        speed = tick_data['speed']
        self._recorder_tick(control) # trjs
        ego_bbox = self.gather_info() # metrics
        self.waypointer.run_step2(ego_bbox + tick_data['offset_pos'], is_gps=True, store=False)
        self.waypointer.show_route()
        if self.save_path is not None and self.step % self.record_every_n_step == 0:
            self.save(control.steer, control.throttle, control.brake, tick_data)
    
    def _lerp(self, v, x):
        D = v.shape[0]

        min_val = self.min_speeds
        max_val = self.max_speeds

        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
        w = x - x0

        return (1-w) * v[x0] + w * v[x1]

    def action_prob(self, steer_logit, throt_logit, brake_logit):

        steer_logit = steer_logit.repeat(self.num_throts)
        throt_logit = throt_logit.repeat_interleave(self.num_steers)

        action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

        return torch.softmax(action_logit, dim=0)

    def post_process(self, steer, throt, brake_prob, spd, cmd):
        
        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake = 0
            throt = max(0.4, throt)

        # # To compensate for non-linearity of throttle<->acceleration
        # if throt > 0.1 and throt < 0.4:
        #     throt = 0.4
        # elif throt < 0.1 and brake_prob > 0.3:
        #     brake = 1

        if spd > {0:10,1:10}.get(cmd, 20)/3.6: # 10 km/h for turning, 15km/h elsewhere
            throt = 0

        # if cmd == 2:
        #     steer = min(max(steer, -0.2), 0.2)

        # if cmd in [4,5]:
        #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing

        return steer, throt, brake
    
def load_state_dict(model, path):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(path)
    
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)


# jxy: mv save in AgentSaver
class AgentSaver(Saver):
    def __init__(self, path_to_conf_file, dict_, list_):
        self.config_path = path_to_conf_file

        # jxy: according to sensor
        self.rgb_list = ['rgb', 'topdown', ] # 'bev', 
        self.add_img = [] # 'flow', 'out', 
        self.lidar_list = [] # 'lidar_0', 'lidar_1',
        self.dir_names = self.rgb_list + self.add_img + self.lidar_list + ['pid_metadata']

        super().__init__(dict_, list_)

    def run(self): # jxy: according to init_ads

        super().run()

    def _save(self, tick_data):    
        # addition
        # save_action_based_measurements = tick_data['save_action_based_measurements']
        self.save_path = tick_data['save_path']
        if not (self.save_path / 'ADS_log.csv' ).exists():
            # addition: generate dir for every total_i
            self.save_path.mkdir(parents=True, exist_ok=True)
            for dir_name in self.dir_names:
                (self.save_path / dir_name).mkdir(parents=True, exist_ok=False)

            # according to self.save data_row_list
            title_row = ','.join(
                ['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake',] + \
                self.dir_names
            )
            with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
                f_out.write(title_row+'\n')

        self.step = tick_data['frame']
        self.save(tick_data['steer'],tick_data['throttle'],tick_data['brake'], tick_data)

    # addition: modified from leaderboard/team_code/auto_pilot.py
    def save(self, steer, throttle, brake, tick_data):
        # frame = self.step // 10
        frame = self.step

        # 'gps' 'thetas'
        pos = tick_data['gps']
        speed = tick_data['speed']
        far_command = tick_data['far_command']
        data_row_list = [frame, far_command.name, speed, steer, throttle, brake,]

        if True: # jxy: according to run_step
            # images
            for rgb_name in self.rgb_list + self.add_img:
                path_ = self.save_path / rgb_name / ('%04d.png' % frame)
                Image.fromarray(tick_data[rgb_name]).save(path_)
                data_row_list.append(str(path_))
            # lidar
            for i, rgb_name in enumerate(self.lidar_list):
                path_ = self.save_path / rgb_name / ('%04d.png' % frame)
                Image.fromarray(matplotlib.cm.gist_earth(tick_data['lidar_processed'][0][0, i], bytes=True)).save(path_)
                data_row_list.append(str(path_))

            # pid_metadata
            pid_metadata = tick_data['pid_metadata']
            path_ = self.save_path / 'pid_metadata' / ('%04d.json' % frame)
            outfile = open(path_, 'w')
            json.dump(pid_metadata, outfile, indent=4)
            outfile.close()
            data_row_list.append(str(path_))

        # collection
        data_row = ','.join([str(i) for i in data_row_list])
        with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
            f_out.write(data_row+'\n')

