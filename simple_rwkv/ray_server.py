from time import sleep
import ray
from simple_rwkv.ray_model import RWKVGenerate, RWKVInfer
from ray.serve.drivers import DAGDriver

inference = RWKVInfer.bind()
generator = RWKVGenerate.bind()

d = DAGDriver.bind({"/inference": inference, "/generator": generator})
