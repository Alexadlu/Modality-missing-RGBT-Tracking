import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
import time


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, checkpoint_path=None, update=[1., 0., 0.]):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, checkpoint_path=checkpoint_path, debug=debug, update=update)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name',nargs='?',default='IPL', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', nargs='?',default='vitb_256_mae_ce_32x4_ep300',type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lashertestingset_miss', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--vis_gpus', type=str, default='5')
    parser.add_argument('--checkpoint_path', type=str, default='./vitb_256_mae_ce_32x4_ep300/IPL_model_ckpt.pth.tar') 
    parser.add_argument('--update', type=float, default=[1.,0.,0.], nargs='+') 
    parser.add_argument('--wait', type=int, default=0)  



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_gpus

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    time.sleep(args.wait*60)
    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, checkpoint_path=args.checkpoint_path, update=args.update)
    # runid参数是无效的,将在tracker中修改为test_epoch


if __name__ == '__main__':
    main()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # run_tracker('ostrack_twobranch', 'vitb_256_mae_ce_32x4_ep300', None, 'rgbt234', None, 1,
    #             0, num_gpus=1)