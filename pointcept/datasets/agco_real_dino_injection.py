import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from .builder import DATASETS
from .defaults import DefaultDataset
import open3d as o3d
import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA 

@DATASETS.register_module()
class AgcoRealDinoDataset(DefaultDataset):
    """
    Loads LiDAR + DINOv3 features (projected from rectified left camera).
    Falls back to geometry-only if `use_dino=False`.
    """

    def __init__(self, ignore_index=-1, use_dino=True, camera="left", **kwargs):
        self.ignore_index = ignore_index
        self.use_dino = use_dino
        self.camera = camera  # "left" or "right"
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self.lidar_topic = "_ouster_points"
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.get_data_list()

        # Camera intrinsics for rectified image
        if camera == "left":
            self.K = np.array([
                [1195.7716, 0.0, 1453.5211],
                [0.0, 1193.8640, 1369.3076],
                [0.0, 0.0, 1.0]
            ])
            
            fov_scale = 9.3
        
            self.K = np.array([
                [94.0477351115933 * fov_scale, 0.0, 1018],
                [0.0, 96.61986494804043* fov_scale, 483],
                [0.0, 0.0, 1.0]
            ])
        else:
            self.K = np.array([
                [1193.6624, 0.0, 1405.9757],
                [0.0, 1192.8856, 1334.0968],
                [0.0, 0.0, 1.0]
            ])


        self.rs_to_cam = rpy_to_matrix(0.0, 0.0, 0.1,-1.572, -0.036, -1.572)
        self.rs_to_ls = rpy_to_matrix(-0.020142, -0.065620, 0.102723, -0.000892, -0.014023,-0.027038)
        self.rs_to_os = rpy_to_matrix(-0.046019, 0.011159, 0.212786,-0.007653 ,-0.007629, 0.001683)
        self.rs_to_rs = rpy_to_matrix(0.0, 0.0, 0.0, 0.0, 0.03, 0.1,)
        
        self.lidar_to_cam_axis = rpy_to_matrix(0.0,0.0,0.0, -1.57, 0.0, -1.572)
        self.rotated_image = rpy_to_matrix(0.0,0.0,0.0,np.pi,0.0,0.0)

        self.angle_adjustment = rpy_to_matrix(0.0, 0.0, 0.0, 0.0, -0.03, 0.0)

        

        # Rectified image resolution (for patch grid)
        self.img_h, self.img_w = 1000, 2000
        self.patch_size = 16

        # Extrinsics (left→right)
        tx, ty, tz = 0.55050726, -0.00634653, -0.00316964
        qx, qy, qz, qw = -0.00032783, 0.00012231, 0.00567948, 0.99998381
        R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [tx, ty, tz]
        self.T_left_right = T

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, (list, tuple)):
            split_list = list(self.split)
        else:
            raise NotImplementedError(f"Unsupported split type: {type(self.split)}")

        data_list = []
        for split in split_list:
            split_dir = os.path.join(self.data_root, split)
            sequences = sorted(os.listdir(split_dir))
            for seq in sequences:
                dino_dir = os.path.join(split_dir,seq,f"_jai_{self.camera}_image_raw/dino_feat")
                coord_dir = os.path.join(split_dir,seq,self.lidar_topic,"coord")
                if not os.path.exists(dino_dir):
                    continue
                for fname in os.listdir(dino_dir):
                    data_list.append(os.path.join(coord_dir, fname))
        self.data_list = data_list
        return data_list

    def get_data(self, idx):
        data_dict = {}
        coord_path = self.data_list[idx % len(self.data_list)]
        dir_path, filename = os.path.split(coord_path)

        data_dict["name"] = self.get_data_name(idx)

        seg_dir = os.path.join(os.path.dirname(dir_path), "segment")
        seg_path = os.path.join(seg_dir, filename)

        coord = np.load(coord_path).astype(np.float32)
        seg = np.load(seg_path).reshape(-1).astype(np.int32)
        seg = np.vectorize(self.learning_map.__getitem__)(seg).astype(np.int32)

        data_dict["coord"] = coord
        data_dict["segment"] = seg


        if self.use_dino:
            seq_dir = os.path.dirname(os.path.dirname(os.path.dirname(coord_path)))
            dino_dir = os.path.join(seq_dir, f"_jai_{self.camera}_image_raw/dino_feat")
            image_dir = os.path.join(seq_dir, f"_jai_{self.camera}_image_raw/img_rect")
            dino_file = os.path.join(dino_dir, filename)
            img_file = os.path.join(image_dir, filename)

            if os.path.exists(dino_file):
                feats_2d = np.load(dino_file) # skip CLS token
                feats_3d, valid_mask = self.project_dino_features(coord, feats_2d)
                data_dict['coord'] = coord[valid_mask]
                data_dict['segment'] = seg[valid_mask]

                if False:
                    img_2d = np.load(img_file)
                    plt.imshow(img_2d)
                    plt.show()
                    img_3d = self.project_image_to_points(coord, img_2d)
                    print(img_3d) 
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coord)
                    pcd.colors = o3d.utility.Vector3dVector(img_3d) 

                    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([pcd, lidar_frame])


                # raise RuntimeError("test", feats_3d, feats_2d.shape, feats_3d.shape, coord.shape)
                data_dict["dino_feat"] = feats_3d.astype(np.float32)
                return data_dict

            else:
                raise RuntimeError("no dino feats")
                data_dict["dino_feat"] = np.zeros((coord.shape[0], 1280), dtype=np.float32)


    def project_dino_features(self, points_3d, feats_2d):
        """
        Project LiDAR (Ouster) points into rectified camera plane and sample 2D patch features.
        Uses calibrated extrinsics and intrinsics.
        
        Args:
            points_3d: (N, 3) array of 3D points in Ouster LiDAR frame
            feats_2d: (num_patches, feature_dim) DINO features

        Returns:
            feats_3d: (N, feature_dim) DINO features per point
        """
        
        # --- Build Ouster → Camera transform ---
        # Same as in project_image_to_points
        if self.lidar_topic == "_lslidar_point_cloud":
            T_lidar_cam = np.linalg.inv(self.rs_to_ls) @ self.rs_to_cam @ self.lidar_to_cam_axis @ self.angle_adjustment
        elif self.lidar_topic == "_rslidar_points":
            T_lidar_cam = self.rs_to_rs @ self.rs_to_cam @ self.lidar_to_cam_axis @ self.angle_adjustment
        else: 
            T_lidar_cam = self.rs_to_os @ self.rs_to_cam @ self.lidar_to_cam_axis @ self.angle_adjustment
        
        # --- Apply transform ---
        points_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
        cam_pts = (T_lidar_cam @ points_h.T).T[:, :3]
        
        # --- Project using intrinsics ---
        uvw = (self.K @ cam_pts.T).T
        
        # Filter points behind camera
        valid_depth = cam_pts[:, 2] > 0.1
        uv = np.zeros_like(uvw[:, :2])
        uv[valid_depth] = uvw[valid_depth, :2] / uvw[valid_depth, 2:3]
        
        # --- Process DINO features ---
        # feats_2d should be (num_patches, feature_dim) after CLS removal
        num_patches, feature_dim = feats_2d.shape
        
        # Calculate actual grid size from number of patches
        grid_h = int(np.sqrt(num_patches * self.img_h / self.img_w))
        grid_w = num_patches // grid_h
        
        # Verify
        if grid_h * grid_w != num_patches:
            # Try to find best factorization
            for h in range(1, num_patches + 1):
                if num_patches % h == 0:
                    w = num_patches // h
                    if abs(h / w - self.img_h / self.img_w) < 0.1:  # aspect ratio check
                        grid_h, grid_w = h, w
                        break
        
        assert grid_h * grid_w == num_patches, \
            f"Cannot reshape {num_patches} patches into grid. Got {grid_h}×{grid_w}={grid_h*grid_w}"
        
        feat_map = feats_2d.reshape(grid_h, grid_w, feature_dim)
        
        # --- Map pixel coordinates to patch coordinates ---
        # DINO patch coordinates based on actual grid size
        u_idx = np.clip((uv[:, 0] * grid_w / self.img_w).astype(int), 0, grid_w - 1)
        v_idx = np.clip((uv[:, 1] * grid_h / self.img_h).astype(int), 0, grid_h - 1)
        
        # Sample features
        feats_3d = feat_map[v_idx, u_idx]
        
        # Optional: Zero out features for invalid points (behind camera or out of bounds)
        img_h, img_w = self.img_h, self.img_w
        valid_mask = (
            valid_depth &
            (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        )
        
        # Set invalid point features to zero
        feats_3d[~valid_mask] = 0.0
        
        cam_pts = cam_pts[valid_mask]
        feats_3d = feats_3d[valid_mask]
        
        # pca = PCA(n_components=3)
        #
        # pca.fit(feats_3d)
        # dino_colors = pca.transform(feats_3d)
        # colors_2d = pca.transform(feats_2d).reshape(grid_h, grid_w, 3)
        # plt.imshow(colors_2d)
        # plt.show()
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cam_pts)
        # pcd.colors = o3d.utility.Vector3dVector(dino_colors)
        # o3d.visualization.draw_geometries([pcd])
        
        


        return feats_3d, valid_mask 
    def project_image_to_points(self, points_3d, image_2d):
        """
        Project LiDAR (Ouster) points into image plane and sample RGB colors.
        Uses calibrated extrinsics and intrinsics.
        
        Args:
            points_3d: (N, 3) array of 3D points in Ouster LiDAR frame
            image_2d: (H, W, 3) rectified RGB image

        Returns:
            colors_3d: (N, 3) RGB color per point
        """

        # --- Build Ouster → Camera transform ---
        # Ouster → RSLidar → Camera

        if self.lidar_topic == "_lslidar_point_cloud":
            T_lidar_cam = np.linalg.inv(self.rs_to_ls) @self.rs_to_cam @ self.lidar_to_cam_axis @ self.angle_adjustment
        elif self.lidar_topic =="_rslidar_points":
            T_lidar_cam = self.rs_to_rs @ self.rs_to_cam @ self.lidar_to_cam_axis @ self.angle_adjustment
        else: 
            T_lidar_cam = self.rs_to_os @self.rs_to_cam @ self.lidar_to_cam_axis @ self.angle_adjustment
        # --- Apply transform ---
        points_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
        cam_pts = (T_lidar_cam @ points_h.T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cam_pts)
        lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # --- Filter points behind camera ---
        valid_depth = cam_pts[:, 2] > 0.1  # 10 cm in front
        print(f"Valid forward points: {valid_depth.sum()} / {len(valid_depth)}")

        # --- Project using intrinsics ---
        uvw = (self.K @ cam_pts.T).T
        uv = np.zeros_like(uvw[:, :2])
        uv[valid_depth] = uvw[valid_depth, :2] / uvw[valid_depth, 2:3]

        # --- Image bounds ---
        img_h, img_w = image_2d.shape[:2]
        valid_mask = (
            valid_depth &
            (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        )

        print(f"Valid points in image bounds: {valid_mask.sum()} / {len(valid_mask)}")
        print(f"UV range: u[{uv[:,0].min():.1f}, {uv[:,0].max():.1f}], v[{uv[:,1].min():.1f}, {uv[:,1].max():.1f}]")

        # --- Sample colors ---
        colors_3d = np.ones((len(points_3d), 3), dtype=np.float32) * 0.5
        u_idx = np.clip(uv[:, 0].astype(int), 0, img_w - 1)
        v_idx = np.clip(uv[:, 1].astype(int), 0, img_h - 1)
        colors_3d[valid_mask] = image_2d[v_idx[valid_mask], u_idx[valid_mask]]
        
        colors_3d[valid_mask] = colors_3d[valid_mask] / 255 
        pcd.colors = o3d.utility.Vector3dVector(colors_3d)

        o3d.visualization.draw_geometries([pcd, lidar_frame])
        return colors_3d

    @staticmethod
    def get_learning_map(ignore_index):
        return {ignore_index: ignore_index, **{i: i for i in range(8)}}

    @staticmethod
    def get_learning_map_inv(ignore_index):
        return {ignore_index: ignore_index, **{i: i for i in range(8)}}




def rpy_to_matrix(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])
    T[:3, :3] = (Rz @ Ry @ Rx)
    return T
