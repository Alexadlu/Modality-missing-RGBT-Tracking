from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.



    # multi modality
    settings.gtot_path = '../GTOT/'
    settings.lasher_path = '../LasHeR/'
    settings.lashertestingSet_path = '../LasHeR/'
    settings.vtuav_path = '../VTUAV/'

    settings.rgbt210_path = '../RGBT210/'
    settings.rgbt234_path = '../RGBT234/'

    settings.result_plot_path = '../test/result_plots'
    settings.results_path = '../tracking_results'    # Where to store tracking results


    settings.save_dir = '..'
    settings.network_path = '..'  

    return settings

