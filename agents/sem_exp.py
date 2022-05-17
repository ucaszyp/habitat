import math
import os

import cv2
import envs.utils.pose as pu
import numpy as np
import skimage.morphology
from constants import color_palette
from envs.habitat.nav import NavRLEnv
from envs.habitat.objectgoal_env import ObjectGoal_Env
from envs.utils.fmm_planner import FMMPlanner
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
from torchvision import transforms
import click

import agents.utils.visualization as vu
from agents.utils.semantic_prediction import SemanticPredMaskRCNN


class Sem_Exp_Env_Agent(NavRLEnv):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):
        self.rank = rank
        self.args = args
        super().__init__(args, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args, rank)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.counter = 0
        self.outroot = "/mnt/yupeng/OGN/sample_debug"

        self.scene_id = None
        self.scene_path = None
        self.scene = None
        self.scene_keys = {}
        self.scene_height = {}
        self.repeat_height = None
        self.cur_eps_id = None
        self.goal_name = None
        self.stg = None

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
            self.sim_vis = None
            self.sim_gt = None

    def reset(self):
        args = self.args
        obs, info = super().reset()
        self.scene_id = self._env.current_episode.scene_id.split("/")[-1]
        self.scene_path = "/".join(self._env.current_episode.scene_id.split("/")[0: -1])
        self.scene = self.scene_id.split(".")[-3]
        self.cur_eps_id = int(self._env.current_episode.episode_id)



        if self.scene not in self.scene_keys:
            self.scene_keys[self.scene] = 0
        self.scene_keys[self.scene] = 0

        height = self._env.current_episode.start_position[1]
        self.repeat_height = 0
        if self.scene not in self.scene_height:
            self.scene_height[self.scene] = []
            
        if height not in self.scene_height[self.scene]:
            self.scene_height[self.scene].append(height)

        elif height in self.scene_height[self.scene]:
            self.repeat_height = 1

        obs = self._preprocess_obs(obs)
        self.goal_name = info["goal_name"]
        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        # if (self._env.current_episode.episode_id == "003688" or self.cur_eps_id == 30020) and self.scene == "1S7LAXRdDqK":
        #     print("begin debug step")
        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), 0., False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0
        if self.args.player == 1:
            print("Please input action")
            key = click.getchar()
            print(key)
            if key == "w":
                action = 1
            if key == "a":
                action = 2
            if key == "d":
                action = 3
        loc = 0
        if self.args.random == 1:
            # if self.repeat_height == 1:
            #     action = 0
            # else:
            #     action = np.random.randint(1, 4)
            action = 3
        loc = self._env.sim.get_agent_state(0)
        
        if self.args.player == 0 and self.args.random == 0:    
            action = self._plan(planner_inputs)

        
        if self.args.visualize or self.args.print_images:
            # if self.scene_keys[self.scene] % 25 == 0:
            self._visualize(planner_inputs, loc)
        self.scene_keys[self.scene] += 1
        if action >= 0:

            # act
            action = {'action': action}
            obs, rew, done, info = super().step(action)

            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = action['action']
            # if self.args.sem_gt == 1:
            #     rgb = obs[0:3, :, :]
            #     depth = obs[3:4, :, :]
            #     obs = np.concatenate((rgb, depth), axis=2)
            self.obs = obs
            self.info = info

            info['g_reward'] += rew

            return obs, rew, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), 0., False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.map_resolution - gx1),
                          int(c * 100.0 / args.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)
        self.stg = stg

        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1])) #arctan() DEG
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            angle_file = "/mnt/yupeng/OGN/vis/vis_full_test1/1UnKg1rAb8A/angle.txt"
            with open(angle_file, "a") as f:
                angle_str = str(relative_angle) + '\n'
                f.write(angle_str)

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        save_tra = 0
        if save_tra == 1:
            counter = self.scene_keys[self.scene]
            tra_dir = "./debug/tra"
            if not os.path.exists(tra_dir):
                os.makedirs(tra_dir)
            tra_file = os.path.join(tra_dir, "%04d" %counter + ".png") 
            cv2.imwrite(tra_file, traversible*255)
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0

        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1
        
        if save_tra == 1:
            counter = self.scene_keys[self.scene]
            tra_dir = "./debug/tra_all"
            if not os.path.exists(tra_dir):
                os.makedirs(tra_dir)
            tra_file = os.path.join(tra_dir, "%04d" %counter + ".png") 
            cv2.imwrite(tra_file, traversible*255)

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        semantic = obs[:, :, 4:5]
        sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), use_seg=use_seg)
        self.sem_gt = semantic
        if self.args.sem_gt == 1:    
            rank = self.rank
            counter = self.scene_keys[self.scene] / 25

            rgb_root = os.path.join(self.outroot, "rgb", self.scene, '%06d' %self.cur_eps_id)
            sem_root = os.path.join(self.outroot, "sem", self.scene, '%06d' %self.cur_eps_id)
            if not os.path.exists(rgb_root):
                os.makedirs(rgb_root)
            if not os.path.exists(sem_root):
                os.makedirs(sem_root)
            rgb_path = os.path.join(rgb_root, '%05d' %counter + '.png')
            sem_path = os.path.join(sem_root, '%05d' %counter + '.png')
            if self.scene_keys[self.scene] % 25 == 0:
                sem = semantic.astype(np.uint8) 
                cv2.imwrite(sem_path, sem)
                rgb = rgb[:, :, ::-1]
                rgb = rgb.astype(np.uint8)
                cv2.imwrite(rgb_path, rgb)


            
        
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)
        
        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            sem_pred = semantic_pred.astype(np.uint8)
            h, w, c = semantic_pred.shape
            sem = np.zeros((h, w), dtype=np.uint8)
            # for i in range(1):
            #     sem += (255 - (sem_pred[:, :, i] * i * 16 + 15))
            sem = sem_pred[:, :, 0] * 255
            self.sem_vis = sem
            semantic_pred = semantic_pred.astype(np.float32)
            

        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred

    def _visualize(self, inputs, loc):
        args = self.args
        # dump_dir = "{}/dump/{}/".format(args.dump_location,
        #                                 args.exp_name)
        # ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
        #     dump_dir, self.rank, self.episode_no)
        dump_dir = "/DATA2/habitat/check1"
        dump_dir1 = "/DATA2/vis_np1"
        ep_dir = '{}/{}/{}/'.format(
              dump_dir, self.scene, '%06d' %self.cur_eps_id)
        ep_dir1 = '{}/{}/{}/'.format(
              dump_dir1, self.scene, '%06d' %self.cur_eps_id)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        if not os.path.exists(ep_dir1):
            os.makedirs(ep_dir1)
        counter = self.scene_keys[self.scene]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']
        
        if self.args.get_bird:
            full_map = inputs["full"]
            full_sem = full_map[4:, :, :].argmax(0)
            full_map_pred = full_map[0, :, :]
            full_exp_pred = full_map[1, :, :]
            
            full_sem += 5
            full_no_cat_mask = full_sem == self.args.classes + 5
            full_map_mask = np.rint(full_map_pred) == 1
            full_exp_mask = np.rint(full_exp_pred) == 1
            # full_vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

            full_sem[full_no_cat_mask] = 0
            m1 = np.logical_and(full_no_cat_mask, full_exp_mask)
            full_sem[m1] = 2

            m2 = np.logical_and(full_no_cat_mask, full_map_mask)
            full_sem[m2] = 1

            # full_sem[full_vis_mask] = 3
            color_pal = [int(x * 255.) for x in color_palette]
            full_sem_map_vis = Image.new("P", (full_sem.shape[1],
                                      full_sem.shape[0]))
            full_sem_map_vis.putpalette(color_pal)
            full_sem_map_vis.putdata(full_sem.flatten().astype(np.uint8))
            full_sem_map_vis = full_sem_map_vis.convert("RGB")
            full_sem_map_vis = np.flipud(full_sem_map_vis)

            full_sem_map_vis = full_sem_map_vis[:, :, [2, 1, 0]]
            full_sem_map_vis = cv2.resize(full_sem_map_vis, (960, 960),
                                 interpolation=cv2.INTER_NEAREST)
            
            ffn = '{}/{}/{}/{}.png'.format(
                dump_dir, self.scene, '%06d' %self.cur_eps_id,
                '%06d' %counter + "full")
            cv2.imwrite(ffn, full_sem_map_vis)

        # gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        if counter == 180:
            print("yes!!!!!!!!!")

        no_cat_mask = sem_map == self.args.classes + 5
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3
  

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        (stg_x, stg_y) = self.stg
        stg_x = round(stg_x)
        stg_y = round(stg_y)
        sem_map[stg_x][stg_y] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        self.sem_vis = np.expand_dims(self.sem_vis, axis=-1)
        self.sem_vis = np.concatenate((self.sem_vis, self.sem_vis, self.sem_vis), axis=-1)
        self.sem_gt = np.concatenate((self.sem_gt, self.sem_gt, self.sem_gt), axis=-1)
        self.vis_image[540:1020, 15:655] = self.sem_vis
        self.vis_image[540:1020, 670:1310] = self.sem_gt

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        #cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2
        y = str(loc.position[0])
        z = str(loc.position[1])
        x = str(loc.position[2])
        # print(x, y, z)
        # print(self._env.current_episode.start_position)
        text = "loc:({},{},{})".format(y, z, x)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (3150 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        self.vis_image = cv2.putText(self.vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)
        
        if args.print_images:
            fn = '{}/{}/{}/{}.png'.format(
                dump_dir, self.scene, '%06d' %self.cur_eps_id,
                '%06d' %counter)
            cv2.imwrite(fn, self.vis_image)
            loc_file = '{}/{}/{}/loc.txt'.format(
                dump_dir, self.scene, '%06d' %self.cur_eps_id)
            with open(loc_file, "w") as f:
                loc_str = "{}, {}, {}".format(y, z, x)
                f.write(loc_str)
        if args.save_np:
            fnn = '{}/{}/{}/{}.npy'.format(
                dump_dir1, self.scene, '%06d' %self.cur_eps_id,
                '%06d' %counter)
            np.save(fnn, self.sem_gt[:,:,0])
        

        self.vis_image[0:50,1120:1920] = 255

if __name__ == '__main__':
    a = Sem_Exp_Env_Agent()