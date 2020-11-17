from time import time

import numpy as np

from cuda_optim import utils
from cuda_optim.cct345_data_generator import list_cct_data_generate, list_qs_data_generate
from cuda_optim.cuda_code import cuda_particle_run_parallel
from cuda_optim.utils import cuda_data_running_particles_at_second_part_entry, analyze_and_output, \
    cuda_particle_data_to_running_particle

momentum_dispersions = [0.0, 0.05]
particle_number_per_plane_per_dp = 4

if __name__ == '__main__':
    start_time = time()

    # 读取 input.txt 的全部机架配置
    params: np.ndarray = utils.read_param()

    # 机架数目
    gantry_number = params.shape[0]
    print(f"gantry_number={gantry_number}")

    # cuda 可读的 cct 参数
    cct_data = list_cct_data_generate(params)

    # cuda 可读的 qs 参数
    qs_data = list_qs_data_generate(params)

    # 生成 cuda 可读的粒子数据
    particle_data = cuda_data_running_particles_at_second_part_entry(
        momentum_dispersion_list=momentum_dispersions, particle_number=particle_number_per_plane_per_dp)

    # 真实运行的粒子数
    particle_number = np.array([int(particle_data.shape[0] / 9)], np.int32)  # 指针
    # 总粒子数
    total_particle_number = particle_number[0] * gantry_number

    print(f"particle_number_per_plane_per_dp={particle_number_per_plane_per_dp}")
    print(f"particle_number_per_gantry={particle_number[0]}")
    print(f"total_particle_number={total_particle_number}")

    # 开辟空间，存储 cuda 返回值
    result = np.empty((total_particle_number * 6,), dtype=np.float32)

    # cuda run
    cuda_particle_run_parallel(gantry_number, cct_data, qs_data, particle_number, particle_data, result)

    # 把 cuda 返回的 result 转为 running_particle_list_list
    running_particle_list_list = cuda_particle_data_to_running_particle(result, gantry_number, particle_number[0])

    # 分析 running_particle_list_list 输入结果到 output.txt
    analyze_and_output(running_particle_list_list, gantry_number, total_particle_number, momentum_dispersions)

    end_time = time()
    print(f"用时{end_time - start_time}s")
