class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '../../../'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.vtuav_dir = '../VTUAV/'
        self.rgbt210_dir = '../RGBT210/' 
        self.rgbt234_dir = '../RGBT234/'
        self.gtot_dir = '../GTOT/'
        self.lasher_trainingset_dir = '../LasHeR'
        self.lasher_testingset_dir = '../LasHeR'