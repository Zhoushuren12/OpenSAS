import numpy as np
import openseespy.opensees as ops
import time

def PushoverAnalysis(
        CtrlNodes: list, story_heights: list,
        Dmax: float, Dincr_init: float, maxRunTime: float,
        ShowAnimation: bool,
        min_factor: float=1e-6, max_factor: float=1.0
        ):

    CtrlNode = CtrlNodes[-1]
    
    # ==================================================
    # 1. 系统初始化 (采用最佳实践)
    # ==================================================
    ops.wipeAnalysis()
    
    # [优化1]: 优先使用 UmfPack 加速，没有则回退
    try:
        ops.system("UmfPack")
    except:
        ops.system("BandGeneral")

    # [优化2]: Transformation 约束比 Plain 更稳健（适合带刚域/刚性隔板的模型）
    ops.constraints("Transformation") 
    ops.numberer("RCM")
    ops.test("EnergyIncr", 1e-6, 100)
    ops.algorithm("KrylovNewton") 
    ops.integrator("DisplacementControl", CtrlNode, 1, Dincr_init)
    ops.analysis("Static")

    # ==================================================
    # 2. 定义算法救赎库 (按优先级排序)
    # ==================================================
    # 逻辑：标准牛顿 -> 线搜索 -> 拟牛顿法 -> 放宽容差
    recovery_algorithms = [
        ("NewtonLineSearch", ['Bisection'], "EnergyIncr", 1e-5, 100),
        ("NewtonLineSearch", ['Secant'],    "EnergyIncr", 1e-5, 100),
        ("ModifiedNewton",   [],            "EnergyIncr", 1e-4, 500),
        ("KrylovNewton",     [],            "NormDispIncr", 1e-4, 500), # 绝招：放宽判敛标准
        ("BFGS",             [],            "EnergyIncr", 1e-4, 500)
    ]

    # ==================================================
    # 3. 变量初始化 (严格保留你要求的格式)
    # ==================================================
    current_disp = 0.0
    factor = 1.0
    Dincr = Dincr_init
    start_time = time.time()

    # --- [用户指定保留区 START] ---
    num_stories = len(story_heights)
    SDRs_list = [[0.0] * num_stories for _ in range(10)]
    SDR_roof = [0.0] * 10
    # --- [用户指定保留区 END] ---

    print(f"--- Pushover Start --- Target: {Dmax}, Dincr: {Dincr}")

    # ==================================================
    # 4. 分析主循环
    # ==================================================
    while current_disp < Dmax:
        
        # 4.1 超时检查
        if time.time() - start_time >= maxRunTime:
            print("Time Limit Exceeded.")
            return 3, current_disp, np.array(SDRs_list), SDR_roof

        # 4.2 执行分析
        ok = ops.analyze(1)

        # ----------------------
        # A. 成功
        # ----------------------
        if ok == 0:
            current_disp = ops.nodeDisp(CtrlNode, 1)
            
            # 记录数据 (使用 append 比 vstack 快得多)
            sdr_i, sdr_roof_i = get_SDR(CtrlNodes, story_heights)
            SDRs_list.append(sdr_i)
            SDR_roof.append(sdr_roof_i)

            # 步长自适应：成功后适度放大，加快计算
            if factor < max_factor:
                factor = min(factor * 1.5, max_factor)
                Dincr = Dincr_init * factor
                ops.integrator("DisplacementControl", CtrlNode, 1, Dincr)
            
            # 恢复主算法
            ops.algorithm("KrylovNewton")
            ops.test("EnergyIncr", 1e-6, 100)

        # ----------------------
        # B. 失败 - 救赎逻辑 (Code A 与 Code B 的结合)
        # ----------------------
        else:
            print(f"!!! Convergence fail at {current_disp:.4f}. Recovering...")
            solved = False
            
            # B.1 优先尝试切换算法 (不减小步长)
            for alg_name, alg_args, test_type, test_tol, test_iter in recovery_algorithms:
                ops.algorithm(alg_name, *alg_args)
                ops.test(test_type, test_tol, test_iter)
                
                if ops.analyze(1) == 0:
                    solved = True
                    print(f"--> Recovered using {alg_name}")
                    break 
            
            # B.2 如果算法全失败，则减小步长
            if not solved:
                factor *= 0.5
                print(f"--> Reducing step factor to {factor:.6f}")
                
                if factor < min_factor:
                    print("Error: Minimum step size reached. Analysis Failed.")
                    return 2, current_disp, np.array(SDRs_list), SDR_roof
                
                Dincr = Dincr_init * factor
                ops.integrator("DisplacementControl", CtrlNode, 1, Dincr)
                
                # 减小步长后，用最稳的算法重试
                ops.algorithm("NewtonLineSearch", "Bisection")
                ops.test("EnergyIncr", 1e-4, 200)
            
            else:
                # 救赎成功：补录数据
                current_disp = ops.nodeDisp(CtrlNode, 1)
                sdr_i, sdr_roof_i = get_SDR(CtrlNodes, story_heights)
                SDRs_list.append(sdr_i)
                SDR_roof.append(sdr_roof_i)
                # 注意：救赎成功这步通常不立即放大步长，求稳

    print("Pushover Analysis Completed Successfully.")
    return 1, current_disp, np.array(SDRs_list), SDR_roof

def get_SDR(ctrlNodes, story_heights):
    """
    辅助函数：计算层间位移角
    """
    SDRs = []
    # 明确指明获取 DOF 1 (X方向) 的位移，防止返回列表格式错误
    for i, h in enumerate(story_heights):
        if i == 0:
            disp_b = 0.0
            disp_t = ops.nodeDisp(ctrlNodes[0], 1)
        else:
            disp_b = ops.nodeDisp(ctrlNodes[i - 1], 1)
            disp_t = ops.nodeDisp(ctrlNodes[i], 1)
        
        SDR = (disp_t - disp_b) / h
        SDRs.append(SDR)
    
    disp_roof = ops.nodeDisp(ctrlNodes[-1], 1)
    # 防止除以零
    total_h = sum(story_heights)
    SDR_roof = disp_roof / total_h if total_h != 0 else 0.0
    
    return SDRs, SDR_roof