"""
Utils to start and stop a CARLA server.
Code adapted from https://github.com/zhejz/carla-roach
"""
import subprocess
import time
from omegaconf import OmegaConf
# os.environ.get('CUDA_VISIBLE_DEVICES')

import logging

log = logging.getLogger(__name__)


def kill_carla():
  with subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True) as kill_process:
    kill_process.wait()
  time.sleep(1)
  log.info('Kill Carla Servers!')


class CarlaServerManager():
  """
  Starts and stops a CARLA server
  """

  def __init__(self, carla_sh_str, port=2000, configs=None, t_sleep=5):
    self._carla_sh_str = carla_sh_str
    # self._root_save_dir = root_save_dir
    self._t_sleep = t_sleep
    self.env_configs = []

    if configs is None:
      cfg = {
          'gpu': 0,
          'port': port,
      }
      self.env_configs.append(cfg)
    else:
      for cfg in configs:
        for gpu in cfg['gpu']:
          single_env_cfg = OmegaConf.to_container(cfg)
          single_env_cfg['gpu'] = gpu
          single_env_cfg['port'] = port
          self.env_configs.append(single_env_cfg)
          port += 5

  def start(self):
    kill_carla()
    for cfg in self.env_configs:
      cmd = f'CUDA_VISIBLE_DEVICES={cfg["gpu"]} bash {self._carla_sh_str} ' \
          f'-fps=10 -carla-rpc-port={cfg["port"]} -RenderOffScreen -nosound &'  # -quality-level=Epic
      #     f'-fps=10 -carla-server -opengl -carla-rpc-port={cfg["port"]}'
      log.info(cmd)
      # log_file = self._root_save_dir / f'server_{cfg["port"]}.log'
      # server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, stdout=open(log_file, "w"))
      with subprocess.Popen(cmd, shell=True, start_new_session=True):
        pass
    time.sleep(self._t_sleep)

  def stop(self):
    kill_carla()
    time.sleep(self._t_sleep)
    log.info('Kill Carla Servers!')
