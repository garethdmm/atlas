"""
This script finds all the results objects from a long run on a remote machine and moves
them to the local machine for easier analysis.
"""

import importlib
import os
import subprocess
import sys

import ml.infra.work_spec


def get_list_of_source_destination_pairs(work_spec):
    source_destination_pairs = []

    remote_az_dir = os.environ['REMOTE_AZ_DIR']
    local_az_dir = os.environ['LOCAL_AZ_DIR']

    relative_file_paths = work_spec.get_all_results_obj_paths()
    relative_dir_paths = work_spec.get_all_model_dirs()

    remote_file_paths = [remote_az_dir + p for p in relative_file_paths]
    local_dir_paths = [local_az_dir + p for p in relative_dir_paths]

    for (file_path, dir_path) in zip(remote_file_paths, local_dir_paths):
        source_destination_pairs.append((file_path, dir_path))

    return source_destination_pairs


def gather_results_from_remote_machine(host_str, source_destination_pairs, execute):
    """
    Given the (remote_file_path, local_destination_directory) pairs, scp each of those
    files onto the local machine.
    """

    for (source_file, destination_dir) in source_destination_pairs:
            try:
                os.makedirs(destination_dir)
            except Exception as e:
                # TODO: handle gracefully if this directory exists but is empty.
                print e

            source = '%s:%s' % (host_str, source_file)

            # TODO: Continue if one of these fails for some reason.
            if execute is True:
                p = subprocess.Popen([
                    'scp',
                    source,
                    destination_dir,
                ])

                sts = os.waitpid(p.pid, 0)


def main():
    work_spec_name = sys.argv[1]
    host_str = sys.argv[2]
    execute = (sys.argv[3] == 'True')

    work_spec = ml.infra.work_spec.WorkSpec.get_work_spec_object_by_name(work_spec_name)

    source_destination_pairs = get_list_of_source_destination_pairs(work_spec)

    gather_results_from_remote_machine(
        host_str,
        source_destination_pairs,
        execute,
    )


if __name__ == '__main__':
    main()

