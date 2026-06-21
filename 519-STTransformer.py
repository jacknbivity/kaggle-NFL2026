import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool as MultiprocessingPool, cpu_count
from tqdm.auto import tqdm
import pickle
import os
import sys
import random
import joblib
from pathlib import Path
import json
import datetime
import logging
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ============ 日志设置 ============
def setup_logging():
    """
    设置日志系统：
    1. 在 log 目录下创建以当前时间命名的子文件夹
    2. 在该子文件夹内保存运行日志和脚本备份
    3. 记录所有运行日志（同时输出到控制台和文件）
    """
    # 创建 log 根目录
    log_root = Path("./log")
    log_root.mkdir(exist_ok=True)
    
    # 生成时间戳并创建对应的子文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_root / timestamp
    run_dir.mkdir(exist_ok=True)
    
    # 日志文件路径
    log_filename = run_dir / "run.log"
    
    # 复制当前脚本到子文件夹作为备份
    script_path = Path(__file__) if '__file__' in globals() else Path("./besttmp.py")
    script_backup = run_dir / "besttmp.py"
    try:
        shutil.copy2(script_path, script_backup)
    except Exception as e:
        # 初始化时无法使用logger，暂时跳过
        pass
    
    # 配置日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 清除现有处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录初始化信息
    logger.info("=" * 80)
    logger.info(f"Logging initialized")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Code backup: {script_backup}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Working Directory: {Path.cwd()}")
    logger.info("=" * 80)
    
    return logger, log_filename

# 初始化日志系统
logger, LOG_FILE = setup_logging()

# ============ 常量定义 ============
FIELD_LENGTH, FIELD_WIDTH = 120.0, 53.3

# ============ 固定全局随机种子 ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
logger.info(f"Global random seed set to {SEED}")
# =========================================

# ============ 配置 ============
# basedir = '/kaggle/input/nfl-big-data-bowl-2026-prediction'
basedir = './nfl-big-data-bowl-2026-prediction'

# 支持命令行参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], 
                    help='Run mode: train or infer')
parser.add_argument('--model_path', type=str, default='./saved_models_rnn',
                    help='Path to save/load models')
parser.add_argument('--use_cache', action='store_true', default=True,
                    help='Use cached sequences if available (default: True)')
parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                    help='Force rebuild sequences (ignore cache)')
args, _ = parser.parse_known_args()

MODE = args.mode  # 'train' 或 'infer'
MODEL_SAVE_PATH = args.model_path  # RNN模型保存路径
USE_CACHE = args.use_cache  # 是否使用缓存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RNN训练配置
N_FOLDS = 10
SEEDS = [42]
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 1e-3
WINDOW_SIZE = 8
HIDDEN_DIM = 128
MAX_FUTURE_HORIZON = 94
USE_SPATIOTEMPORAL = True  # 🚀 启用时空Transformer ([B,T,N,D]格式，更强空间建模)
K_NEIGHBORS = 3  # 时空模型的邻居数

# 数据质量控制
MAX_REASONABLE_OUTPUT_FRAMES = 50  # 截断超过5秒的输出（过滤异常长的plays）

# 内存优化
USE_FLOAT32 = True
POLARS_FLOAT_TYPE = pl.Float32 if USE_FLOAT32 else pl.Float64
NUMPY_FLOAT_TYPE = np.float32 if USE_FLOAT32 else np.float64
logger.info(f"Memory optimization: {'float32 (50% memory saved)' if USE_FLOAT32 else 'float64 (default)'}")
logger.info(f"Device: {DEVICE}")
# ===================================

# ============ 方向统一处理函数 ============
def unify_left_direction(df):
    """
    Mirror rightward plays so all samples are 'left' oriented (x,y, dir, o, ball_land).
    将所有右向进攻统一为左向，便于特征学习；保留原方向到 play_direction_orig
    """
    if 'play_direction' not in df.columns:
        return df
    
    # 保存原方向（如果还没保存过）
    if 'play_direction_orig' not in df.columns:
        df = df.with_columns([
            pl.col('play_direction').alias('play_direction_orig')
        ])
    
    # 使用 Polars 表达式（不是 Series）
    right_mask = (pl.col('play_direction') == 'right')
    
    # 位置镜像
    if 'x' in df.columns:
        df = df.with_columns([
            pl.when(right_mask).then(FIELD_LENGTH - pl.col('x')).otherwise(pl.col('x')).alias('x')
        ])
    if 'y' in df.columns:
        df = df.with_columns([
            pl.when(right_mask).then(FIELD_WIDTH - pl.col('y')).otherwise(pl.col('y')).alias('y')
        ])
    
    # 角度镜像 (dir, o)
    for col in ('dir', 'o'):
        if col in df.columns:
            df = df.with_columns([
                pl.when(right_mask).then((pl.col(col) + 180.0) % 360.0).otherwise(pl.col(col)).alias(col)
            ])
    
    # 球落点镜像
    if 'ball_land_x' in df.columns:
        df = df.with_columns([
            pl.when(right_mask).then(FIELD_LENGTH - pl.col('ball_land_x')).otherwise(pl.col('ball_land_x')).alias('ball_land_x')
        ])
    if 'ball_land_y' in df.columns:
        df = df.with_columns([
            pl.when(right_mask).then(FIELD_WIDTH - pl.col('ball_land_y')).otherwise(pl.col('ball_land_y')).alias('ball_land_y')
        ])
    
    # 统一方向列为 'left'，避免后续特征再按原方向做二次处理
    df = df.with_columns([
        pl.lit('left').alias('play_direction')
    ])
    
    return df

def invert_to_original_direction(x_u, y_u, play_dir_right: bool):
    """
    Invert unified (left) coordinates back to original play direction.
    将统一后的左向坐标转换回原始方向
    """
    if not play_dir_right:
        return float(x_u), float(y_u)
    return float(FIELD_LENGTH - x_u), float(FIELD_WIDTH - y_u)

def build_play_direction_map(df):
    """
    构建 (game_id, play_id) -> play_direction 的映射
    """
    direction_map = (
        df.select(['game_id', 'play_id', 'play_direction'])
        .unique()
        .to_pandas()
        .set_index(['game_id', 'play_id'])['play_direction']
    )
    return direction_map
    
def load_weekly_data(week_num):
    input_df = pl.read_csv(f'{basedir}/train/input_2023_w{week_num:02d}.csv')
    output_df = pl.read_csv(f'{basedir}/train/output_2023_w{week_num:02d}.csv')
    return input_df, output_df

def load_all_train_data():
    logger.info("Loading training data...")
    with MultiprocessingPool(min(cpu_count(), 18)) as pool:
        results = list(tqdm(pool.imap(load_weekly_data, range(1, 19)), total=18))
    
    input_dfs = [r[0] for r in results]
    output_dfs = [r[1] for r in results]
    
    input_data = pl.concat(input_dfs)
    output_data = pl.concat(output_dfs)
    
    # 🚀 内存优化：转换浮点列为float32
    if USE_FLOAT32:
        float_cols = [col for col in input_data.columns if input_data[col].dtype == pl.Float64]
        if float_cols:
            input_data = input_data.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
            logger.info(f"  ✅ Converted {len(float_cols)} float64 columns to float32 in input_data")
        
        float_cols = [col for col in output_data.columns if output_data[col].dtype == pl.Float64]
        if float_cols:
            output_data = output_data.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
            logger.info(f"  ✅ Converted {len(float_cols)} float64 columns to float32 in output_data")
    
    logger.info(f"Input data shape: {input_data.shape}")
    logger.info(f"Output data shape: {output_data.shape}")
    
    return input_data, output_data

def engineer_advanced_features(df):
    """Advanced feature engineering with sequence and interaction features - Polars version"""
    import gc
    
    df = df.with_columns([
        # Velocity components
        (pl.col('s') * pl.col('dir').radians().cos()).alias('velocity_x'),
        (pl.col('s') * pl.col('dir').radians().sin()).alias('velocity_y'),
    ])
    
    df = df.with_columns([
        # Distance and angle to ball
        ((pl.col('x') - pl.col('ball_land_x'))**2 + (pl.col('y') - pl.col('ball_land_y'))**2).sqrt().alias('dist_to_ball'),
        pl.arctan2(pl.col('ball_land_y') - pl.col('y'), pl.col('ball_land_x') - pl.col('x')).alias('angle_to_ball'),
    ])
    
    df = df.with_columns([
        # Velocity toward ball
        (pl.col('velocity_x') * pl.col('angle_to_ball').cos() + 
         pl.col('velocity_y') * pl.col('angle_to_ball').sin()).alias('velocity_toward_ball'),
        
        # Time to ball
        (pl.col('num_frames_output') / 10.0).alias('time_to_ball'),
        
        # Orientation diff
        (pl.col('o') - pl.col('dir')).abs().alias('orientation_diff_temp'),
    ])
    
    df = df.with_columns([
        pl.min_horizontal([pl.col('orientation_diff_temp'), 360 - pl.col('orientation_diff_temp')]).alias('orientation_diff')
    ]).drop('orientation_diff_temp')
    
    df = df.with_columns([
        # Role features
        (pl.col('player_role') == 'Targeted Receiver').cast(pl.Int32).alias('role_targeted_receiver'),
        (pl.col('player_role') == 'Defensive Coverage').cast(pl.Int32).alias('role_defensive_coverage'),
        (pl.col('player_role') == 'Passer').cast(pl.Int32).alias('role_passer'),
        (pl.col('player_side') == 'Offense').cast(pl.Int32).alias('side_offense'),
    ])
    
    # Height and BMI
    df = df.with_columns([
        pl.col('player_height').str.split('-').list.get(0).cast(pl.Float64).alias('height_feet'),
        pl.col('player_height').str.split('-').list.get(1).cast(pl.Float64).alias('height_inch_part'),
    ])
    
    df = df.with_columns([
        (pl.col('height_feet') * 12 + pl.col('height_inch_part')).alias('height_inches'),
    ]).drop(['height_feet', 'height_inch_part'])
    
    df = df.with_columns([
        ((pl.col('player_weight') / (pl.col('height_inches')**2)) * 703).alias('bmi'),
    ])
    
    df = df.with_columns([
        # Acceleration components
        (pl.col('a') * pl.col('dir').radians().cos()).alias('acceleration_x'),
        (pl.col('a') * pl.col('dir').radians().sin()).alias('acceleration_y'),
        
        # Distance to target
        (pl.col('ball_land_x') - pl.col('x')).alias('distance_to_target_x'),
        (pl.col('ball_land_y') - pl.col('y')).alias('distance_to_target_y'),
        
        # Speed squared
        (pl.col('s') ** 2).alias('speed_squared'),
    ])
    
    df = df.with_columns([
        (pl.col('acceleration_x')**2 + pl.col('acceleration_y')**2).sqrt().alias('accel_magnitude'),
        (pl.col('angle_to_ball') - pl.col('dir').radians()).cos().alias('velocity_alignment'),
    ])
    
    df = df.with_columns([
        # Expected position at ball
        (pl.col('x') + pl.col('velocity_x') * pl.col('time_to_ball')).alias('expected_x_at_ball'),
        (pl.col('y') + pl.col('velocity_y') * pl.col('time_to_ball')).alias('expected_y_at_ball'),
    ])
    
    df = df.with_columns([
        # Error from ball
        (pl.col('expected_x_at_ball') - pl.col('ball_land_x')).alias('error_from_ball_x'),
        (pl.col('expected_y_at_ball') - pl.col('ball_land_y')).alias('error_from_ball_y'),
    ])
    
    df = df.with_columns([
        (pl.col('error_from_ball_x')**2 + pl.col('error_from_ball_y')**2).sqrt().alias('error_from_ball'),
    ])
    
    df = df.with_columns([
        # Momentum
        (pl.col('player_weight') * pl.col('velocity_x')).alias('momentum_x'),
        (pl.col('player_weight') * pl.col('velocity_y')).alias('momentum_y'),
        
        # Kinetic energy
        (0.5 * pl.col('player_weight') * pl.col('speed_squared')).alias('kinetic_energy'),
    ])
    
    df = df.with_columns([
        (pl.col('o') - pl.col('angle_to_ball').degrees()).abs().alias('angle_diff_temp'),
    ])
    
    df = df.with_columns([
        pl.min_horizontal([pl.col('angle_diff_temp'), 360 - pl.col('angle_diff_temp')]).alias('angle_diff')
    ]).drop('angle_diff_temp')
    
    df = df.with_columns([
        (pl.col('time_to_ball') ** 2).alias('time_squared'),
        (pl.col('dist_to_ball') ** 2).alias('dist_squared'),
        (pl.col('dist_to_ball') / (pl.col('time_to_ball') + 0.1)).alias('weighted_dist_by_time'),
    ])
    
    # === SPATIAL AND FIELD POSITION FEATURES ===
    df = df.with_columns([
        (pl.col('x') / 120.0).alias('field_position_x'),
        (pl.col('y') / 53.3).alias('field_position_y'),
        ((pl.col('x') - 60)**2 + (pl.col('y') - 26.65)**2).sqrt().alias('field_center_distance'),
    ])
    
    df = df.with_columns([
        ((pl.col('x') < 10) | (pl.col('x') > 110)).cast(pl.Int32).alias('in_endzone'),
        ((pl.col('x') >= 50) & (pl.col('x') <= 70)).cast(pl.Int32).alias('in_midfield'),
        (((pl.col('x') >= 0) & (pl.col('x') <= 20)) | ((pl.col('x') >= 100) & (pl.col('x') <= 120))).cast(pl.Int32).alias('in_redzone'),
        pl.when(pl.col('x') <= 60).then(0).otherwise(1).alias('field_side'),
    ])
    
    df = df.with_columns([
        pl.min_horizontal([pl.col('y'), 53.3 - pl.col('y')]).alias('distance_to_sideline'),
        pl.min_horizontal([pl.col('x'), 120 - pl.col('x')]).alias('distance_to_goal_line'),
        pl.min_horizontal([pl.col('x'), 120 - pl.col('x')]).alias('distance_to_endzone'),
    ])
    
    df = df.with_columns([
        (pl.col('distance_to_sideline') < 5).cast(pl.Int32).alias('near_sideline'),
        (pl.col('distance_to_goal_line') < 10).cast(pl.Int32).alias('near_goal_line'),
        (pl.col('field_center_distance') < 10).cast(pl.Int32).alias('near_center'),
    ])
    
    # Field quadrant
    df = df.with_columns([
        pl.when((pl.col('x') <= 60) & (pl.col('y') <= 26.65))
            .then(1)
            .when((pl.col('x') > 60) & (pl.col('y') <= 26.65))
            .then(2)
            .when((pl.col('x') <= 60) & (pl.col('y') > 26.65))
            .then(3)
            .otherwise(4)
            .alias('field_quadrant'),
    ])
    
    df = df.with_columns([
        (pl.col('play_direction') == 'right').cast(pl.Int32).alias('play_direction_right'),
    ])
    
    df = df.with_columns([
        pl.when(pl.col('play_direction_right') == 1)
            .then(pl.col('x'))
            .otherwise(120 - pl.col('x'))
            .alias('relative_x_position'),
    ])
    
    df = df.with_columns([
        pl.col('absolute_yardline_number').alias('yardline_position'),
        (pl.col('absolute_yardline_number') / 100.0).alias('yardline_normalized'),
    ])
    
    df = df.with_columns([
        pl.min_horizontal([pl.col('yardline_position'), 100 - pl.col('yardline_position')]).alias('distance_to_goal'),
    ])
    
    df = df.with_columns([
        ((pl.col('distance_to_sideline') < 3) | 
         (pl.col('distance_to_goal_line') < 5) |
         (pl.col('field_center_distance') < 5)).cast(pl.Int32).alias('in_pressure_zone'),
    ])
    
    # 释放内存
    gc.collect()
    
    return df

def add_sequence_features(df):
    """Add temporal lag and rolling features - Polars version"""
    import gc
    
    df = df.sort(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    group_cols = ['game_id', 'play_id', 'nfl_id']
    
    # Lag features
    lag_exprs = []
    for lag in [1, 2, 3, 4, 5, 7]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            lag_exprs.append(
                pl.col(col).shift(lag).over(group_cols).alias(f'{col}_lag{lag}')
            )
    
    df = df.with_columns(lag_exprs)
    
    # Rolling features
    rolling_exprs = []
    for window in [3, 5, 7, 10]:
        for col in ['x', 'y','velocity_x', 'velocity_y', 's']:
            rolling_exprs.extend([
                pl.col(col).rolling_mean(window_size=window, min_periods=1).over(group_cols).alias(f'{col}_rolling_mean_{window}'),
                pl.col(col).rolling_std(window_size=window, min_periods=1).over(group_cols).alias(f'{col}_rolling_std_{window}'),
            ])
    
    df = df.with_columns(rolling_exprs)
    
    trend_exprs = []
    for col in ['s', 'velocity_x', 'velocity_y']:
        # First and second differences
        trend_exprs.extend([
            (pl.col(col) - pl.col(col).shift(1).over(group_cols)).alias(f'{col}_diff1'),
        ])
    
    df = df.with_columns(trend_exprs)
    
    # Second diff needs first diff to exist
    trend_exprs2 = []
    for col in ['s', 'velocity_x', 'velocity_y']:
        trend_exprs2.extend([
            (pl.col(f'{col}_diff1') - pl.col(f'{col}_diff1').shift(1).over(group_cols)).alias(f'{col}_diff2'),
            ((pl.col(col) - pl.col(col).shift(3).over(group_cols)) / 3.0).alias(f'{col}_trend_3'),
            ((pl.col(col) - pl.col(col).shift(5).over(group_cols)) / 5.0).alias(f'{col}_trend_5'),
            ((pl.col(col) - pl.col(col).shift(10).over(group_cols)) / 10.0).alias(f'{col}_trend_10'),
        ])
    
    df = df.with_columns(trend_exprs2)
    
    # Trend accel
    trend_accel_exprs = []
    for col in ['s','velocity_x', 'velocity_y']:
        trend_accel_exprs.append(
            (pl.col(f'{col}_trend_3') - pl.col(f'{col}_trend_3').shift(1).over(group_cols)).alias(f'{col}_trend_accel')
        )
    
    df = df.with_columns(trend_accel_exprs)
    
    # Velocity delta
    delta_exprs = []
    for col in ['velocity_x', 'velocity_y']:
        delta_exprs.append(
            (pl.col(col) - pl.col(col).shift(1).over(group_cols)).alias(f'{col}_delta')
        )
    
    df = df.with_columns(delta_exprs)
    
    # 释放内存
    del lag_exprs, rolling_exprs, trend_exprs, trend_exprs2, trend_accel_exprs, delta_exprs
    gc.collect()
    
    return df

def add_trajectory_and_prediction_features(df):
    """添加轨迹曲率、预测准确性、路径效率和拦截概率特征 - Polars version"""
    import gc
    
    df = df.sort(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    group_cols = ['game_id', 'play_id', 'nfl_id']
    
    # === 1. 轨迹预测特征 ===
    df = df.with_columns([
        (pl.col('x') + pl.col('velocity_x') * 0.1).alias('predicted_x_linear'),
        (pl.col('y') + pl.col('velocity_y') * 0.1).alias('predicted_y_linear'),
        (pl.col('x') + pl.col('velocity_x') * 0.1 + 0.5 * pl.col('acceleration_x') * (0.1**2)).alias('predicted_x_accel'),
        (pl.col('y') + pl.col('velocity_y') * 0.1 + 0.5 * pl.col('acceleration_y') * (0.1**2)).alias('predicted_y_accel'),
    ])
    
    # === 2. 预测准确性指标 ===
    df = df.with_columns([
        pl.col('predicted_x_linear').shift(1).over(group_cols).alias('prev_predicted_x'),
        pl.col('predicted_y_linear').shift(1).over(group_cols).alias('prev_predicted_y'),
    ])
    
    df = df.with_columns([
        (pl.col('x') - pl.col('prev_predicted_x')).abs().alias('prediction_error_x'),
        (pl.col('y') - pl.col('prev_predicted_y')).abs().alias('prediction_error_y'),
    ])
    
    # ⚡ 删除临时列释放内存
    df = df.drop(['prev_predicted_x', 'prev_predicted_y'])
    
    df = df.with_columns([
        (pl.col('prediction_error_x')**2 + pl.col('prediction_error_y')**2).sqrt().alias('prediction_error_total'),
    ])
    
    df = df.with_columns([
        pl.col('prediction_error_total').rolling_mean(window_size=3, min_periods=1).over(group_cols).alias('prediction_error_rolling_mean'),
    ])
    
    # Velocity change rate
    df = df.with_columns([
        pl.when((pl.col('velocity_x_delta').is_not_null()) & (pl.col('velocity_y_delta').is_not_null()))
            .then((pl.col('velocity_x_delta')**2 + pl.col('velocity_y_delta')**2).sqrt())
            .otherwise(0)
            .alias('velocity_change_rate')
    ])
    
    # === 3. 轨迹曲率特征 ===
    df = df.with_columns([
        pl.col('dir').shift(1).over(group_cols).alias('dir_lag1'),
    ])
    
    df = df.with_columns([
        (pl.col('dir') - pl.col('dir_lag1')).alias('dir_change_temp'),
    ])
    
    # ⚡ 删除 dir_lag1
    df = df.drop('dir_lag1')
    
    df = df.with_columns([
        pl.when(pl.col('dir_change_temp') > 180)
            .then(pl.col('dir_change_temp') - 360)
            .when(pl.col('dir_change_temp') < -180)
            .then(pl.col('dir_change_temp') + 360)
            .otherwise(pl.col('dir_change_temp'))
            .alias('dir_change'),
    ])
    
    df = df.with_columns([
        pl.col('dir_change').abs().alias('dir_change_abs'),
    ])
    
    # Curvature calculation - 合并计算减少内存
    df = df.with_columns([
        pl.col('x').shift(1).over(group_cols).alias('x_lag1'),
        pl.col('y').shift(1).over(group_cols).alias('y_lag1'),
        pl.col('x').shift(2).over(group_cols).alias('x_lag2'),
        pl.col('y').shift(2).over(group_cols).alias('y_lag2'),
    ])
    
    df = df.with_columns([
        (pl.col('x_lag1') - pl.col('x_lag2')).alias('vec1_x'),
        (pl.col('y_lag1') - pl.col('y_lag2')).alias('vec1_y'),
        (pl.col('x') - pl.col('x_lag1')).alias('vec2_x'),
        (pl.col('y') - pl.col('y_lag1')).alias('vec2_y'),
    ])
    
    # 删除lag列
    df = df.drop(['x_lag2', 'y_lag2'])
    
    df = df.with_columns([
        (pl.col('vec1_x') * pl.col('vec2_y') - pl.col('vec1_y') * pl.col('vec2_x')).alias('cross_product'),
        (pl.col('vec1_x') * pl.col('vec2_x') + pl.col('vec1_y') * pl.col('vec2_y')).alias('dot_product'),
    ])
    
    df = df.with_columns([
        pl.arctan2(pl.col('cross_product'), pl.col('dot_product') + 1e-10).alias('trajectory_curvature'),
    ])
    
    df = df.with_columns([
        pl.col('trajectory_curvature').abs().alias('trajectory_curvature_abs'),
    ])
    
    df = df.with_columns([
        pl.col('trajectory_curvature_abs').rolling_mean(window_size=3, min_periods=1).over(group_cols).alias('curvature_rolling_mean'),
        pl.col('trajectory_curvature_abs').rolling_std(window_size=3, min_periods=1).over(group_cols).alias('curvature_rolling_std'),
    ])
    
    df = df.with_columns([
        (pl.col('dir_change_abs') > 30).cast(pl.Int32).alias('sharp_turn'),
    ])
    
    # === 5. 拦截概率特征 ===
    df = df.with_columns([
        (pl.col('dist_to_ball') / (pl.col('time_to_ball') + 0.01)).alias('interception_speed_needed'),
    ])
    
    df = df.with_columns([
        (pl.col('s') / (pl.col('interception_speed_needed') + 0.01)).alias('speed_ratio_to_intercept'),
    ])
    
    df = df.with_columns([
        (pl.col('angle_to_ball') - pl.col('dir').radians()).cos().alias('velocity_alignment_to_ball'),
    ])
    
    df = df.with_columns([
        (pl.col('speed_ratio_to_intercept') * (pl.col('velocity_alignment_to_ball') + 1) / 2).clip(0, 1).alias('interception_feasibility'),
    ])
    
    df = df.with_columns([
        (pl.col('interception_feasibility') * (
            pl.col('role_targeted_receiver') * 1.5 +
            pl.col('role_defensive_coverage') * 1.2 +
            (1 - pl.col('role_targeted_receiver') - pl.col('role_defensive_coverage')) * 0.5
        )).alias('interception_prob_weighted'),
    ])
    
    df = df.with_columns([
        (pl.col('speed_ratio_to_intercept') > 0.8).cast(pl.Int32).alias('can_intercept_in_time'),
    ])
    
    # Intercept angle deviation
    df = df.with_columns([
        pl.arctan2(pl.col('velocity_y'), pl.col('velocity_x')).degrees().alias('current_angle'),
        pl.col('angle_to_ball').degrees().alias('optimal_angle'),
    ])
    
    df = df.with_columns([
        (pl.col('current_angle') - pl.col('optimal_angle')).abs().alias('intercept_angle_deviation_temp'),
    ])
    
    df = df.with_columns([
        pl.min_horizontal([pl.col('intercept_angle_deviation_temp'), 360 - pl.col('intercept_angle_deviation_temp')]).alias('intercept_angle_deviation')
    ])
    
    # ⚠️ 修改：移除 path_efficiency（未使用特征），重新调整权重
    df = df.with_columns([
        ((1 - pl.col('intercept_angle_deviation') / 180) * 0.5 +
         pl.col('speed_ratio_to_intercept').clip(0, 1) * 0.5).alias('interception_efficiency'),
    ])
    
    # Relative rankings
    df = df.with_columns([
        pl.col('dist_to_ball').rank(method='average').over(['game_id', 'play_id', 'frame_id']).alias('relative_dist_to_ball_rank'),
        pl.col('velocity_toward_ball').rank(method='average').over(['game_id', 'play_id', 'frame_id']).alias('relative_speed_to_ball_rank'),
    ])
    
    # Convert ranks to percentiles
    df = df.with_columns([
        (pl.col('relative_dist_to_ball_rank') / pl.col('relative_dist_to_ball_rank').max().over(['game_id', 'play_id', 'frame_id'])).alias('relative_dist_to_ball'),
        (pl.col('relative_speed_to_ball_rank') / pl.col('relative_speed_to_ball_rank').max().over(['game_id', 'play_id', 'frame_id'])).alias('relative_speed_to_ball'),
    ])
    
    df = df.with_columns([
        ((1 - pl.col('relative_dist_to_ball')) * 0.6 + pl.col('relative_speed_to_ball') * 0.4).alias('interception_advantage'),
    ])
    
    # Drop temporary columns
    temp_cols = ['dir_change_temp', 'vec1_x', 'vec1_y', 'vec2_x', 'vec2_y', 
                 'cross_product', 'dot_product', 'current_angle', 'optimal_angle',
                 'intercept_angle_deviation_temp', 'relative_dist_to_ball_rank', 
                 'relative_speed_to_ball_rank', 'x_lag1', 'y_lag1',
                 'step_distance', 'cumulative_distance', 'start_x', 'start_y', 'straight_line_distance']
    df = df.drop([c for c in temp_cols if c in df.columns])
    
    # 释放内存
    del temp_cols
    gc.collect()
    
    return df

def add_physics_constraint_features(df):
    """添加物理约束特征 - 基于运动学和动力学原理 - Polars version"""
    import gc
    
    df = df.sort(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    group_cols = ['game_id', 'play_id', 'nfl_id']
    
    # === 1. 方向与位移的物理一致性 ===    
    # 根据方向角计算理论上的位移分量
    df = df.with_columns([
        # 方向角对应的单位向量
        pl.col('dir').radians().sin().alias('dir_unit_x'),  # NFL坐标系：dir=0指向y轴正方向
        pl.col('dir').radians().cos().alias('dir_unit_y'),
    ])
    
    # 计算实际位移
    df = df.with_columns([
        (pl.col('x') - pl.col('x').shift(1).over(group_cols)).alias('actual_dx'),
        (pl.col('y') - pl.col('y').shift(1).over(group_cols)).alias('actual_dy'),
    ])
    
    # 理论位移（基于速度和方向）
    df = df.with_columns([
        (pl.col('s') * 0.1 * pl.col('dir_unit_x')).alias('expected_dx'),  # 0.1秒时间步长
        (pl.col('s') * 0.1 * pl.col('dir_unit_y')).alias('expected_dy'),
    ])
    
    # 位移一致性指标
    df = df.with_columns([
        # 实际位移与理论位移的差异
        (pl.col('actual_dx') - pl.col('expected_dx')).abs().alias('displacement_error_x'),
        (pl.col('actual_dy') - pl.col('expected_dy')).abs().alias('displacement_error_y'),
    ])
    
    df = df.with_columns([
        # 总位移误差
        ((pl.col('displacement_error_x')**2 + pl.col('displacement_error_y')**2).sqrt()).alias('displacement_error_total'),
    ])
    
    df = df.with_columns([
        # 位移一致性得分 (0-1, 1表示完全一致)
        (1.0 / (1.0 + pl.col('displacement_error_total'))).alias('displacement_consistency_score'),
    ])
    
    # === 2. 运动方向特征 ===
    
    # 实际运动方向（基于位移）
    df = df.with_columns([
        pl.arctan2(pl.col('actual_dy'), pl.col('actual_dx')).degrees().alias('actual_motion_dir'),
    ])
    
    # 方向偏差
    df = df.with_columns([
        (pl.col('actual_motion_dir') - pl.col('dir')).alias('dir_deviation_temp'),
    ])
    
    df = df.with_columns([
        # 标准化到[-180, 180]
        pl.when(pl.col('dir_deviation_temp') > 180)
            .then(pl.col('dir_deviation_temp') - 360)
            .when(pl.col('dir_deviation_temp') < -180)
            .then(pl.col('dir_deviation_temp') + 360)
            .otherwise(pl.col('dir_deviation_temp'))
            .alias('direction_deviation'),
    ])
    
    df = df.with_columns([
        pl.col('direction_deviation').abs().alias('direction_deviation_abs'),
    ])
    
    # === 4. 加速度物理约束 ===    
    # 理论最大加速度（基于人体物理极限，NFL球员约10 m/s²）
    df = df.with_columns([
        pl.lit(10.0).alias('max_human_acceleration'),  # m/s²
    ])
    
    # === 7. 能量守恒相关特征 ===    
    # 动能变化
    df = df.with_columns([
        pl.col('kinetic_energy').shift(1).over(group_cols).alias('prev_kinetic_energy'),
    ])
    
    # 清理临时列
    temp_cols = ['dir_unit_x', 'dir_unit_y', 'expected_dx', 'expected_dy',
                 'dir_deviation_temp', 'max_human_acceleration', 'prev_kinetic_energy']
                 # 以下列已在特征创建时被注释，不需要删除：
                 # 'expected_speed_change', 'actual_speed_change', 
                 # 'x_axis_dominance', 'y_axis_dominance', 'actual_xy_ratio', 'expected_xy_ratio'
    df = df.drop([c for c in temp_cols if c in df.columns])
    
    # 释放内存
    del temp_cols
    gc.collect()
    
    return df

def add_football_rule_features(df):
    """添加橄榄球规则和战术相关特征 - Polars version"""
    import gc
    
    # === 1. 接球竞争特征 ===    
    keys = ['game_id', 'play_id']
    
    df = df.with_columns([
        # 接球点附近的防守/进攻球员
        pl.when((pl.col('dist_to_ball') <= 5) & (pl.col('player_side') == 'Defense'))
            .then(1)
            .otherwise(0)
            .alias('_def_at_catch'),
        pl.when((pl.col('dist_to_ball') <= 5) & (pl.col('player_side') == 'Offense'))
            .then(1)
            .otherwise(0)
            .alias('_off_at_catch'),
    ])
    
    df = df.with_columns([
        pl.col('_def_at_catch').sum().over(keys).alias('defenders_at_catch_point'),
        pl.col('_off_at_catch').sum().over(keys).alias('receivers_at_catch_point'),
    ])
    
    df = df.with_columns([
        # 接球点优势
        (pl.col('receivers_at_catch_point') - pl.col('defenders_at_catch_point')).alias('catch_point_advantage'),
        
        # 是否为竞争接球
        ((pl.col('receivers_at_catch_point') + pl.col('defenders_at_catch_point')) >= 3)
            .cast(pl.Int32)
            .alias('is_contested_catch'),
    ])
    
    # === 2. 红区和关键区域特征 ===    
    df = df.with_columns([
        # 红区特征
        ((pl.col('yardline_position') <= 20) | (pl.col('yardline_position') >= 80))
            .cast(pl.Int32)
            .alias('in_redzone_pass'),
        
        ((pl.col('yardline_position') <= 10) | (pl.col('yardline_position') >= 90))
            .cast(pl.Int32)
            .alias('near_goalline_pass'),
        
        ((pl.col('yardline_position') <= 2) | (pl.col('yardline_position') >= 98))
            .cast(pl.Int32)
            .alias('two_point_zone'),
        
        # 到中场距离
        pl.when(pl.col('yardline_position') < 50)
            .then(50 - pl.col('yardline_position'))
            .otherwise(pl.col('yardline_position') - 50)
            .alias('yards_to_midfield'),
    ])
    
    df = df.with_columns([
        # 红区拥挤度
        pl.when(pl.col('in_redzone_pass') == 1)
            .then(pl.col('gnn_ally_cnt') + pl.col('gnn_opp_cnt'))
            .otherwise(0)
            .alias('redzone_congestion'),
        
        # 端区垂直约束
        pl.when(pl.col('near_goalline_pass') == 1)
            .then(pl.min_horizontal([pl.col('distance_to_goal_line'), 10.0]))
            .otherwise(10.0)
            .alias('endzone_vertical_constraint'),
    ])
    # 清理临时列
    df = df.drop(['_def_at_catch', '_off_at_catch'])
    
    # 释放内存
    gc.collect()
    
    return df

def compute_neighbor_embeddings_per_frame(input_df, k_neigh=6, radius=30.0, tau=8.0):
    """
    Per-frame GNN-lite neighbor embeddings: time-varying player interactions.
    Computes spatial features for each frame, enabling temporal modeling of spatial dynamics.
    """
    import gc
    
    cols_needed = ["game_id","play_id","frame_id","nfl_id","x","y",
                   "velocity_x","velocity_y","player_side"]
    src = input_df.select([c for c in cols_needed if c in input_df.columns])

    # Self-join: find neighbors within the same (game, play, frame)
    tmp = src.join(
        src.rename({
            "frame_id":"frame_id_nb",
            "nfl_id":"nfl_id_nb",
            "x":"x_nb", "y":"y_nb",
            "velocity_x":"vx_nb", "velocity_y":"vy_nb",
            "player_side":"player_side_nb"
        }),
        left_on=["game_id","play_id","frame_id"],
        right_on=["game_id","play_id","frame_id_nb"],
        how="left",
    )

    # Drop self
    tmp = tmp.filter(pl.col("nfl_id_nb") != pl.col("nfl_id"))

    # Relative vectors and distance
    tmp = tmp.with_columns([
        (pl.col("x_nb") - pl.col("x")).alias("dx"),
        (pl.col("y_nb") - pl.col("y")).alias("dy"),
        (pl.col("vx_nb") - pl.col("velocity_x")).alias("dvx"),
        (pl.col("vy_nb") - pl.col("velocity_y")).alias("dvy"),
    ])
    tmp = tmp.with_columns([(pl.col("dx")**2 + pl.col("dy")**2).sqrt().alias("dist")])
    tmp = tmp.filter(pl.col("dist").is_finite() & (pl.col("dist") > 1e-6))
    if radius is not None:
        tmp = tmp.filter(pl.col("dist") <= float(radius))

    # Ally / opponent flag
    tmp = tmp.with_columns([
        (pl.col("player_side_nb").fill_null("") == pl.col("player_side").fill_null("")).cast(pl.Float32).alias("is_ally")
    ])

    # Rank by distance (keep top-K per frame per player)
    keys = ["game_id","play_id","frame_id","nfl_id"]
    tmp = tmp.with_columns([pl.col("dist").rank(method="ordinal").over(keys).alias("rnk")])
    if k_neigh is not None:
        tmp = tmp.filter(pl.col("rnk") <= float(k_neigh))

    # Attention weights: softmax(-dist/tau) within group
    tmp = tmp.with_columns([(-pl.col("dist") / float(tau)).exp().alias("w")])
    tmp = tmp.with_columns([(pl.col("w") / pl.col("w").sum().over(keys)).fill_null(0.0).alias("wn")])
    tmp = tmp.with_columns([
        (pl.col("wn") * pl.col("is_ally")).alias("wn_ally"),
        (pl.col("wn") * (1.0 - pl.col("is_ally"))).alias("wn_opp"),
    ])

    # Pre-multiply for group sums
    for col in ["dx","dy","dvx","dvy"]:
        tmp = tmp.with_columns([
            (pl.col(col) * pl.col("wn_ally")).alias(f"{col}_ally_w"),
            (pl.col(col) * pl.col("wn_opp")).alias(f"{col}_opp_w"),
        ])

    tmp = tmp.with_columns([
        pl.when(pl.col("is_ally") > 0.5).then(pl.col("dist")).otherwise(None).alias("dist_ally"),
        pl.when(pl.col("is_ally") < 0.5).then(pl.col("dist")).otherwise(None).alias("dist_opp"),
    ])

    ag = tmp.group_by(keys).agg([
        pl.col("dx_ally_w").sum().alias("gnn_ally_dx_mean"),
        pl.col("dy_ally_w").sum().alias("gnn_ally_dy_mean"),
        pl.col("dvx_ally_w").sum().alias("gnn_ally_dvx_mean"),
        pl.col("dvy_ally_w").sum().alias("gnn_ally_dvy_mean"),
        pl.col("dx_opp_w").sum().alias("gnn_opp_dx_mean"),
        pl.col("dy_opp_w").sum().alias("gnn_opp_dy_mean"),
        pl.col("dvx_opp_w").sum().alias("gnn_opp_dvx_mean"),
        pl.col("dvy_opp_w").sum().alias("gnn_opp_dvy_mean"),
        pl.col("is_ally").sum().alias("gnn_ally_cnt"),
        (pl.len() - pl.col("is_ally").sum()).cast(pl.Float64).alias("gnn_opp_cnt"),
        pl.col("dist_ally").min().alias("gnn_ally_dmin"),
        pl.col("dist_ally").mean().alias("gnn_ally_dmean"),
        pl.col("dist_opp").min().alias("gnn_opp_dmin"),
        pl.col("dist_opp").mean().alias("gnn_opp_dmean"),
    ])

    # d1..d3 nearest (regardless of side)
    near = tmp.filter(pl.col("rnk") <= 3).select(keys + ["rnk", "dist"]).with_columns([pl.col("rnk").cast(pl.Int32)])
    d1 = near.filter(pl.col("rnk") == 1).select(keys + [pl.col("dist").alias("gnn_d1")])
    d2 = near.filter(pl.col("rnk") == 2).select(keys + [pl.col("dist").alias("gnn_d2")])
    d3 = near.filter(pl.col("rnk") == 3).select(keys + [pl.col("dist").alias("gnn_d3")])
    ag = ag.join(d1, on=keys, how="left").join(d2, on=keys, how="left").join(d3, on=keys, how="left")

    # Safe fills
    fill_zero = ["gnn_ally_dx_mean","gnn_ally_dy_mean","gnn_ally_dvx_mean","gnn_ally_dvy_mean",
                 "gnn_opp_dx_mean","gnn_opp_dy_mean","gnn_opp_dvx_mean","gnn_opp_dvy_mean",
                 "gnn_ally_cnt","gnn_opp_cnt"]
    for c in fill_zero:
        if c in ag.columns:
            ag = ag.with_columns([pl.col(c).fill_null(0.0)])

    radius_val = radius if radius is not None else 30.0
    for c in ["gnn_ally_dmin","gnn_opp_dmin","gnn_ally_dmean","gnn_opp_dmean","gnn_d1","gnn_d2","gnn_d3"]:
        if c in ag.columns:
            ag = ag.with_columns([pl.col(c).fill_null(radius_val)])

    # 释放内存
    del tmp, near, d1, d2, d3, src
    gc.collect()

    return ag

def add_heading_features(df):
    """Add heading unit vectors (NFL angle convention) - Polars version"""
    
    df = df.with_columns([
        pl.col("dir").fill_null(0.0).radians().sin().alias("heading_x"),
        pl.col("dir").fill_null(0.0).radians().cos().alias("heading_y"),
    ])
    
    return df

def add_time_features(df):
    """添加时间特征 - Polars version"""
    group_cols = ['game_id', 'play_id', 'nfl_id']
    
    df = df.with_columns([
        pl.col('frame_id').cum_count().over(group_cols).alias('frames_elapsed')
    ])
    
    df = df.with_columns([
        (pl.col('frames_elapsed') / (pl.col('frames_elapsed').max().over(group_cols) + 1e-9)).alias('normalized_time')
    ])
    
    df = df.with_columns([
        (2 * np.pi * pl.col('normalized_time')).sin().alias('time_sin'),
        (2 * np.pi * pl.col('normalized_time')).cos().alias('time_cos'),
    ])
    
    return df

def add_qb_relative_features(df):
    # 必要列检查
    need = ['game_id','play_id','frame_id','x','y','velocity_x','velocity_y','dir','player_role']
    if not all(c in df.columns for c in need):
        logger.warning("QB relative features: missing required columns - SKIPPED")
        return df

    # 1) 提取每帧QB坐标（如多QB取first）
    keys = ['game_id','play_id','frame_id']
    qb = (
        df.filter(pl.col('player_role') == 'Passer')
          .group_by(keys, maintain_order=True)
          .agg([
              pl.col('x').first().alias('qb_x'),
              pl.col('y').first().alias('qb_y'),
          ])
    )

    # 2) 连接回原表
    df = df.join(qb, on=keys, how='left')

    # 3) 矢量化计算
    dx = (pl.col('x') - pl.col('qb_x')).cast(pl.Float32)
    dy = (pl.col('y') - pl.col('qb_y')).cast(pl.Float32)
    dist = (dx*dx + dy*dy).sqrt().clip(1e-6, None)  # clip(lower, upper) 兼容旧版

    ux = (dx / dist).alias('_ux')
    uy = (dy / dist).alias('_uy')

    dir_rad = pl.col('dir').fill_null(0.0).radians()
    to_qb_angle = pl.arctan2(-dy, -dx)

    bearing_signed = pl.arctan2(
        (to_qb_angle - dir_rad).sin(),
        (to_qb_angle - dir_rad).cos()
    ).degrees()

    # 第一步：添加基础特征
    df = df.with_columns([
        dist.alias('qb_distance').cast(pl.Float32),

        # 速度在 QB 方向的投影/垂直分量
        (pl.col('velocity_x')*ux + pl.col('velocity_y')*uy)
            .alias('vel_to_qb_alignment').cast(pl.Float32),
        (pl.col('velocity_x')*(-uy) + pl.col('velocity_y')*ux)
            .alias('vel_to_qb_perp').cast(pl.Float32),

        # 朝向 vs 指向QB 的有符号方位差
        bearing_signed.alias('bearing_to_qb_signed').cast(pl.Float32),
    ])
    
    # 第二步：添加 sin/cos 编码（依赖上一步创建的列）
    df = df.with_columns([
        (pl.col('bearing_to_qb_signed') * 3.141592653589793 / 180.0).sin()
            .alias('bearing_to_qb_sin').cast(pl.Float32),
        (pl.col('bearing_to_qb_signed') * 3.141592653589793 / 180.0).cos()
            .alias('bearing_to_qb_cos').cast(pl.Float32),
    ])

    # 4) 清理临时列与可能缺QB的帧（可选：保留NaN便于下游）
    df = df.drop(['qb_x','qb_y'])

    return df


def clean_features_for_modeling(df):
    """
    清理特征数据：处理inf、NaN和超大值
    确保所有数值特征都在合理范围内
    """
    logger.info("Cleaning features: handling inf/nan/extreme values...")
    
    # 获取所有数值列
    numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    
    for col in numeric_cols:
        df = df.with_columns([
            # 1. 替换 inf 为 None
            pl.when(pl.col(col).is_infinite())
              .then(None)
              .otherwise(pl.col(col))
              .alias(col)
        ])
        
        df = df.with_columns([
            # 2. 替换 NaN 为 None
            pl.when(pl.col(col).is_nan())
              .then(None)
              .otherwise(pl.col(col))
              .alias(col)
        ])
        
        df = df.with_columns([
            # 4. 填充剩余的 null 为 0
            pl.col(col).fill_null(0.0).alias(col)
        ])
    
    logger.info("✓ Features cleaned successfully")
    return df


def compute_advanced_graph_features(input_df, k_neigh=6, radius=30.0):
    """
    Compute advanced graph features including multi-hop, Voronoi, tactical, and pressure features
    PLUS deep graph features: directional, velocity-based, path interference, role-specific
    Polars version with optimized performance
    """
    import gc
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y",
                   "velocity_x", "velocity_y", "player_side", "player_role", "dir", "s",
                   "ball_land_x", "ball_land_y", "dist_to_ball", "velocity_toward_ball"]
    src = input_df.select([c for c in cols_needed if c in input_df.columns])
    
    # Get last frame for each player
    last = (src.sort(["game_id", "play_id", "nfl_id", "frame_id"])
               .group_by(["game_id", "play_id", "nfl_id"], maintain_order=True)
               .tail(1)
               .rename({"frame_id": "last_frame_id"}))
    
    # Join neighbors at the ego's last_frame_id
    tmp = last.join(
        src.rename({
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb",
            "velocity_x": "vx_nb", "velocity_y": "vy_nb",
            "player_side": "player_side_nb",
            "player_role": "player_role_nb",
            "dir": "dir_nb", "s": "s_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left",
    )
    
    # Drop self
    tmp = tmp.filter(pl.col("nfl_id_nb") != pl.col("nfl_id"))
    
    # Calculate distances
    tmp = tmp.with_columns([
        (pl.col("x_nb") - pl.col("x")).alias("dx"),
        (pl.col("y_nb") - pl.col("y")).alias("dy"),
    ])
    
    tmp = tmp.with_columns([
        (pl.col("dx")**2 + pl.col("dy")**2).sqrt().alias("dist")
    ])
    
    tmp = tmp.filter(pl.col("dist").is_finite())
    tmp = tmp.filter(pl.col("dist") > 1e-6)
    
    # Identify allies and opponents
    tmp = tmp.with_columns([
        (pl.col("player_side_nb").fill_null("") == pl.col("player_side").fill_null("")).cast(pl.Int32).alias("is_ally"),
        (pl.col("player_role_nb").fill_null("") == "Passer").cast(pl.Int32).alias("is_passer"),
        (pl.col("player_role") == "Targeted Receiver").cast(pl.Int32).alias("is_target_receiver"),
    ])
    
    keys = ["game_id", "play_id", "nfl_id"]
    
    # === 1. MULTI-HOP GRAPH FEATURES ===
    # Hop-1 neighbors (within radius)
    tmp_hop1 = tmp.filter(pl.col("dist") <= radius) if radius else tmp
    
    hop1_stats = tmp_hop1.group_by(keys).agg([
        pl.len().alias("graph_hop1_neighbors"),
    ])
    
    # Extended neighborhood density (neighbors within extended radius)
    # This captures broader spatial context
    tmp_extended = tmp.filter(pl.col("dist") <= radius * 1.5) if radius else tmp
    
    extended_stats = tmp_extended.group_by(keys).agg([
        pl.len().alias("graph_extended_neighbors"),
        pl.col("dist").mean().alias("graph_extended_avg_dist"),
    ])
    
    # Neighbor average degree (count neighbors for each neighbor)
    neighbor_degrees = tmp_hop1.group_by(["game_id", "play_id", "nfl_id_nb"]).agg([
        pl.len().alias("neighbor_degree")
    ])
    
    tmp_with_degree = tmp_hop1.join(
        neighbor_degrees,
        left_on=["game_id", "play_id", "nfl_id_nb"],
        right_on=["game_id", "play_id", "nfl_id_nb"],
        how="left"
    )
    
    neighbor_avg_degree = tmp_with_degree.group_by(keys).agg([
        pl.col("neighbor_degree").mean().fill_null(0).alias("graph_neighbor_avg_degree")
    ])
    
    # Spatial distribution uniformity (coefficient of variation)
    # Lower values = neighbors evenly distributed; higher = clustered in certain directions
    spatial_uniformity = tmp_hop1.group_by(keys).agg([
        (pl.col("dist").std() / (pl.col("dist").mean() + 1e-6)).fill_null(0).alias("graph_spatial_cv")
    ])
    
    # === 2. SPATIAL CONTROL FEATURES ===
    # Use nearest neighbor distances to estimate controlled space
    # More accurate approximation based on k-nearest neighbors
    spatial_control = tmp_hop1.with_columns([
        pl.col("dist").rank(method="ordinal").over(keys).alias("neighbor_rank")
    ]).filter(pl.col("neighbor_rank") <= 3).group_by(keys).agg([
        # Average distance to 3 nearest neighbors as space control indicator
        pl.col("dist").mean().alias("local_space_control"),
        # Minimum distance (closest threat/ally)
        pl.col("dist").min().alias("min_neighbor_dist"),
    ])
    
    # === 3. TACTICAL FEATURES ===
    # Distance to passer
    dist_to_passer = tmp.filter(pl.col("is_passer") == 1).group_by(keys).agg([
        pl.col("dist").min().fill_null(999.0).alias("tactical_dist_to_passer")
    ])
    
    # Distance to targeted receiver (only for defenders)
    # First, get targeted receiver positions
    target_receivers = last.filter(pl.col("player_role").fill_null("") == "Targeted Receiver").select([
        "game_id", "play_id", "x", "y"
    ]).rename({"x": "target_x", "y": "target_y"})
    
    last_with_target = last.join(
        target_receivers,
        on=["game_id", "play_id"],
        how="left"
    )
    
    dist_to_target = last_with_target.with_columns([
        ((pl.col("x") - pl.col("target_x"))**2 + (pl.col("y") - pl.col("target_y"))**2).sqrt().fill_null(999.0).alias("tactical_dist_to_target")
    ]).select(keys + ["tactical_dist_to_target"])
    
    # Defenders nearby (opponents within 5 yards)
    defenders_nearby = tmp.filter((pl.col("is_ally") == 0) & (pl.col("dist") <= 5.0)).group_by(keys).agg([
        pl.len().alias("tactical_defenders_nearby")
    ])
    
    # === 4. PRESSURE FIELD FEATURES ===
    # Pressure based on opponent proximity (inverse distance weighted)
    tmp_pressure = tmp.filter(pl.col("is_ally") == 0).with_columns([
        (1.0 / (pl.col("dist") + 1.0)).alias("pressure_contribution")
    ])
    
    pressure_stats = tmp_pressure.group_by(keys).agg([
        pl.col("pressure_contribution").sum().fill_null(0).alias("pressure_total"),
        pl.col("pressure_contribution").max().fill_null(0).alias("pressure_max_threat"),
        (pl.col("dist") <= 3.0).sum().alias("pressure_close_opponents")
    ])
    # 创建空的占位DataFrame（避免代码后续报错）
    path_interference = last.select(keys).unique()
    local_advantage = last.select(keys).unique()
    catch_competition = last.select(keys).unique()
    neighbor_variance = last.select(keys).unique()
    density_stats = last.select(keys).unique()
    
    
    # === MERGE ALL FEATURES ===
    result = hop1_stats.join(extended_stats, on=keys, how="left")
    result = result.join(neighbor_avg_degree, on=keys, how="left")
    result = result.join(spatial_uniformity, on=keys, how="left")
    result = result.join(spatial_control, on=keys, how="left")
    result = result.join(dist_to_passer, on=keys, how="left")
    result = result.join(dist_to_target, on=keys, how="left")
    result = result.join(defenders_nearby, on=keys, how="left")
    result = result.join(pressure_stats, on=keys, how="left")
    
    # Fill nulls with appropriate defaults
    fill_zero_cols = ["graph_hop1_neighbors", "graph_extended_neighbors", "graph_neighbor_avg_degree",
                      "graph_spatial_cv", "tactical_defenders_nearby", "pressure_total",
                      "pressure_max_threat", "pressure_close_opponents", "graph_extended_avg_dist"]
    
    for col in fill_zero_cols:
        if col in result.columns:
            result = result.with_columns([pl.col(col).fill_null(0.0)])
    
    # Fill spatial control features with reasonable defaults
    if "local_space_control" in result.columns:
        result = result.with_columns([pl.col("local_space_control").fill_null(radius if radius else 30.0)])
    
    if "min_neighbor_dist" in result.columns:
        result = result.with_columns([pl.col("min_neighbor_dist").fill_null(radius if radius else 30.0)])
    
    if "tactical_dist_to_passer" in result.columns:
        result = result.with_columns([pl.col("tactical_dist_to_passer").fill_null(999.0)])
    
    if "tactical_dist_to_target" in result.columns:
        result = result.with_columns([pl.col("tactical_dist_to_target").fill_null(999.0)])
    
    # 释放内存
    del (src, last, tmp, tmp_hop1, tmp_extended, tmp_pressure, hop1_stats, extended_stats, 
         neighbor_degrees, tmp_with_degree, neighbor_avg_degree, spatial_uniformity, 
         spatial_control, dist_to_passer, target_receivers, last_with_target, dist_to_target, 
         defenders_nearby, pressure_stats)
    gc.collect()
    
    return result


# ============ 模型损失函数 ============

class TemporalHuber(nn.Module):
    """来自的损失函数"""
    def __init__(self, delta=0.5, time_decay=0.03, lam_smooth=0.01):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.lam_smooth = lam_smooth

    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta)
        )

        if self.time_decay and self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device, dtype=pred.dtype)
            w = torch.exp(-self.time_decay * t).view(1, L)
            huber = huber * w
            mask  = mask  * w

        main_loss = (huber * mask).sum() / (mask.sum() + 1e-8)

        if self.lam_smooth and pred.size(1) > 2:
            d1 = pred[:, 1:] - pred[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            m2 = mask[:, 2:]
            smooth = (d2 * d2) * m2
            smooth_loss = smooth.sum() / (m2.sum() + 1e-8)
        else:
            smooth_loss = pred.new_tensor(0.0)

        return main_loss + self.lam_smooth * smooth_loss

class STransformer(nn.Module):
    """时空 Transformer - 来自 sttf.py，适配到 besttmp.py 的训练流程
    
    输入: [B, T, N, F] (batch, time, players/nodes, features)
    输出: [B, horizon, 2] (对于单个ego球员的预测) 或 [B, horizon, N, 2] (对于所有球员的预测)
    """
    def __init__(self, input_dim, horizon, hidden_dim=128, n_spatial_layers=2, 
                 n_temporal_layers=2, n_heads=4, dropout=0.1, window_size=8, 
                 max_players=22, predict_single_player=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.window_size = window_size
        self.max_players = max_players
        self.predict_single_player = predict_single_player  # 是否只预测ego球员
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 时间位置编码
        self.temporal_pos_embedding = nn.Embedding(window_size, hidden_dim)
        
        # LayerNorm 用于稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 空间 Transformer (帧内球员交互)
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_encoder_layer, num_layers=n_spatial_layers)
        
        # 时间 Transformer (帧间时序建模)
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=n_temporal_layers)
        
        # 预测头：输出 (x, y) 位移
        self.head = nn.Linear(hidden_dim, horizon * 2)
        
    def forward(self, x, rel_features=None, ego_idx=0, mask=None):
        """
        Args:
            x: [B, T, N, F] 输入特征
            rel_features: 相对特征（保留接口兼容性，本模型不使用）
            ego_idx: ego球员索引（当 predict_single_player=True 时使用）
            mask: 掩码（可选）
            
        Returns:
            如果 predict_single_player=True: [B, horizon, 2]
            否则: [B, horizon, N, 2]
        """
        B, T, N, F = x.shape
        H = self.hidden_dim
        
        # 1. 输入嵌入
        x = self.embedding(x)  # [B, T, N, H]
        
        # 2. 添加时间位置编码（安全处理T可能超出window_size的情况）
        T_actual = min(T, self.window_size)
        temporal_pos = torch.arange(T_actual, device=x.device)  # [T_actual]
        temporal_pe = self.temporal_pos_embedding(temporal_pos)  # [T_actual, H]
        
        # 如果 T > window_size，截断输入；如果 T < window_size，只使用前T个位置编码
        if T > self.window_size:
            x = x[:, :self.window_size, :, :]
            T = self.window_size
        
        temporal_pe = temporal_pe.unsqueeze(0).unsqueeze(2)  # [1, T, 1, H]
        x = x + temporal_pe  # [B, T, N, H]
        
        # 3. LayerNorm
        x = self.layer_norm(x)
        
        # 4. 空间 Transformer (帧内球员交互) - 分批处理避免CUDA错误
        # 重塑为 [B*T, N, H] 以便每帧独立处理
        batch_limit = 512  # 限制批次大小，避免CUDA配置错误
        x_spatial_list = []
        
        for b_start in range(0, B * T, batch_limit):
            b_end = min(b_start + batch_limit, B * T)
            x_chunk = x.reshape(B * T, N, H)[b_start:b_end]
            x_chunk_out = self.spatial_transformer(x_chunk)
            x_spatial_list.append(x_chunk_out)
        
        x_spatial_out = torch.cat(x_spatial_list, dim=0)
        x = x_spatial_out.reshape(B, T, N, H)
        
        # 5. 时间 Transformer (帧间时序建模) - 分批处理避免CUDA错误
        # 重塑为 [B*N, T, H] 以便每个球员独立处理时序
        x_temporal_list = []
        
        for b_start in range(0, B * N, batch_limit):
            b_end = min(b_start + batch_limit, B * N)
            x_chunk = x.permute(0, 2, 1, 3).reshape(B * N, T, H)[b_start:b_end]
            x_chunk_out = self.temporal_transformer(x_chunk)
            x_temporal_list.append(x_chunk_out)
        
        x_temporal_out = torch.cat(x_temporal_list, dim=0)
        
        # 6. 提取最后一个时间步的特征
        last_step_features = x_temporal_out.reshape(B, N, T, H)[:, :, -1, :]  # [B, N, H]
        
        # 7. 预测头：生成轨迹
        preds = self.head(last_step_features)  # [B, N, horizon*2]
        preds = preds.reshape(B, N, self.horizon, 2)  # [B, N, horizon, 2]
        
        # 8. 调整维度 (不进行累积和，直接输出 scaled velocity)
        preds = preds.permute(0, 2, 1, 3)  # [B, horizon, N, 2]
        # preds = torch.cumsum(preds, dim=1)  # [B, horizon, N, 2]
        
        # 9. 根据模式返回
        if self.predict_single_player:
            # 只返回 ego 球员的预测
            return preds[:, :, ego_idx, :]  # [B, horizon, 2]
        else:
            # 返回所有球员的预测
            return preds  # [B, horizon, N, 2]


# ============ 准备序列数据 ============
def prepare_targets(batch_axis, max_h):
    """准备目标张量和掩码"""
    tensors, masks = [], []
    for arr in batch_axis:
        L = len(arr)
        padded = np.pad(arr, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        tensors.append(torch.tensor(padded))
        masks.append(torch.tensor(mask))
    return torch.stack(tensors), torch.stack(masks)


def create_spatiotemporal_subgraphs(input_df, output_df, feature_cols, window_size=10, 
                                     k_neighbors=6, cache_path=None, show_progress=True):
    """
    创建时空子图数据 [B, T, N, D]，N = 1(ego) + K(邻居)
    
    Args:
        show_progress: 是否显示进度条（推理时设为False）
    
    返回:
        node_features: List of [T, N, D] numpy arrays
        rel_features_list: List of dicts containing relative features
        targets_dx, targets_dy: 目标轨迹
        seq_meta: 元数据
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading subgraphs from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            logger.info(f"✓ Loaded {len(cache_data['node_features'])} cached subgraphs")
            return (cache_data['node_features'], cache_data['rel_features_list'],
                    cache_data['targets_dx'], cache_data['targets_dy'], cache_data['seq_meta'])
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, rebuilding...")
    
    logger.info("Creating spatiotemporal subgraphs...")
    input_pd = input_df.to_pandas()
    output_pd = output_df.to_pandas()
    
    idx_x = feature_cols.index('x')
    idx_y = feature_cols.index('y')
    idx_vx = feature_cols.index('velocity_x') if 'velocity_x' in feature_cols else None
    idx_vy = feature_cols.index('velocity_y') if 'velocity_y' in feature_cols else None
    
    node_features = []
    rel_features_list = []
    targets_dx = []
    targets_dy = []
    seq_meta = []
    
    target_groups = output_pd[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    N = k_neighbors + 1  # ego + K neighbors
    
    # 根据 show_progress 决定是否使用 tqdm
    iterator = tqdm(target_groups.iterrows(), total=len(target_groups), desc="Creating subgraphs") if show_progress else target_groups.iterrows()
    
    for _, row in iterator:
        gid, pid, nid = row['game_id'], row['play_id'], row['nfl_id']
        
        # 获取该play的所有球员数据
        play_input = input_pd[
            (input_pd['game_id'] == gid) & (input_pd['play_id'] == pid)
        ].sort_values('frame_id')
        
        if len(play_input) == 0:
            continue
        
        # 获取 ego 球员数据
        ego_input = play_input[play_input['nfl_id'] == nid].copy()
        if len(ego_input) == 0:
            continue
        
        # 取最后 window_size 帧
        frames = ego_input['frame_id'].tail(window_size).values
        if len(frames) < window_size:
            # 填充早期帧
            frames = np.pad(frames, (window_size - len(frames), 0), mode='edge')
        
        # 为每一帧构建 ego + topK 邻居
        frame_nodes = []  # [T, N, D]
        frame_rel_features = {
            'dxy': [],  # [T, N, N, 2]
            'dist': [],
            'angle': [],
            'dv': [],
            'ally_mask': []
        }
        
        for frame_id in frames:
            frame_data = play_input[play_input['frame_id'] == frame_id]
            ego_data = frame_data[frame_data['nfl_id'] == nid]
            
            if len(ego_data) == 0:
                # 使用最近帧的ego数据填充
                ego_data = ego_input.iloc[[-1]]
            
            # 计算到ego的距离，选择topK邻居
            other_players = frame_data[frame_data['nfl_id'] != nid].copy()
            
            if len(other_players) > 0 and 'x' in ego_data.columns:
                ego_x = ego_data.iloc[0]['x']
                ego_y = ego_data.iloc[0]['y']
                other_players['dist_to_ego'] = np.sqrt(
                    (other_players['x'] - ego_x)**2 + 
                    (other_players['y'] - ego_y)**2
                )
                neighbors = other_players.nsmallest(k_neighbors, 'dist_to_ego')
            else:
                neighbors = pd.DataFrame(columns=frame_data.columns)
            
            # 拼接 ego + neighbors
            nodes = pd.concat([ego_data, neighbors], ignore_index=True)
            
            # 填充不足K个邻居的情况
            while len(nodes) < N:
                # 使用padding（全0）
                pad_row = pd.DataFrame([{col: 0 for col in nodes.columns}])
                nodes = pd.concat([nodes, pad_row], ignore_index=True)
            
            nodes = nodes.head(N)  # 确保恰好N个节点
            
            # 提取特征
            try:
                node_feat = nodes[feature_cols].values.astype(np.float32)
            except KeyError:
                node_feat = np.zeros((N, len(feature_cols)), dtype=np.float32)
            
            node_feat = np.nan_to_num(node_feat, nan=0.0)
            frame_nodes.append(node_feat)
            
            # 计算相对特征
            xy = node_feat[:, [idx_x, idx_y]]  # [N, 2]
            dxy = xy[:, None, :] - xy[None, :, :]  # [N, N, 2]
            dist = np.linalg.norm(dxy, axis=-1, keepdims=True)  # [N, N, 1]
            dist = np.clip(dist, 1e-6, None)
            
            angle = np.arctan2(dxy[:, :, 1:2], dxy[:, :, 0:1])  # [N, N, 1]
            
            if idx_vx is not None and idx_vy is not None:
                v = node_feat[:, [idx_vx, idx_vy]]  # [N, 2]
                dv = v[:, None, :] - v[None, :, :]  # [N, N, 2]
            else:
                dv = np.zeros((N, N, 2), dtype=np.float32)
            
            # ally mask (简化：都设为0，实际需要根据player_side判断)
            ally_mask = np.zeros((N, N, 1), dtype=np.float32)
            
            frame_rel_features['dxy'].append(dxy)
            frame_rel_features['dist'].append(dist)
            frame_rel_features['angle'].append(angle)
            frame_rel_features['dv'].append(dv)
            frame_rel_features['ally_mask'].append(ally_mask)
        
        # Stack frames
        seq = np.stack(frame_nodes, axis=0)  # [T, N, D]
        node_features.append(seq)
        
        rel_feats = {
            k: np.stack(v, axis=0) for k, v in frame_rel_features.items()
        }  # Each: [T, N, N, *]
        rel_features_list.append(rel_feats)
        
        # 获取输出
        player_output = output_pd[
            (output_pd['game_id'] == gid) & 
            (output_pd['play_id'] == pid) & 
            (output_pd['nfl_id'] == nid)
        ].sort_values('frame_id')
        
        if len(player_output) == 0:
            node_features.pop()
            rel_features_list.pop()
            continue
        
        # 检查是否有真实坐标（训练模式）还是只有模板（推理模式）
        if 'x' in player_output.columns and 'y' in player_output.columns:
            # 训练模式：计算真实目标
            # 获取绝对坐标序列 (包含未来所有帧)
            future_x = player_output['x'].values
            future_y = player_output['y'].values
            
            # 获取当前帧(t=0)坐标
            last_x = seq[-1, 0, idx_x]
            last_y = seq[-1, 0, idx_y]
            
            # 拼接 [x0, x1, x2, ...]
            full_x = np.concatenate(([last_x], future_x))
            full_y = np.concatenate(([last_y], future_y))
            
            # 计算逐帧差分 (velocity)
            step_dx = full_x[1:] - full_x[:-1]
            step_dy = full_y[1:] - full_y[:-1]
            
            if len(step_dx) > MAX_REASONABLE_OUTPUT_FRAMES:
                step_dx = step_dx[:MAX_REASONABLE_OUTPUT_FRAMES]
                step_dy = step_dy[:MAX_REASONABLE_OUTPUT_FRAMES]
            
            # Scale by 10.0 (matching train.py)
            targets_dx.append((step_dx * 10.0).astype(np.float32))
            targets_dy.append((step_dy * 10.0).astype(np.float32))
        else:
            # 推理模式：使用占位符
            num_output_frames = len(player_output)
            targets_dx.append(np.zeros(num_output_frames, dtype=np.float32))
            targets_dy.append(np.zeros(num_output_frames, dtype=np.float32))
        
        # 元数据
        play_dir = None
        if 'play_direction_orig' in ego_input.columns:
            play_dir = ego_input['play_direction_orig'].iloc[-1]
        
        seq_meta.append({
            'game_id': gid,
            'play_id': pid,
            'nfl_id': nid,
            'play_direction': play_dir,
        })
    
    logger.info(f"✓ Created {len(node_features)} subgraphs, N={N}, T={window_size}")
    
    # 缓存
    if cache_path:
        logger.info(f"Saving subgraphs to cache: {cache_path}")
        cache_data = {
            'node_features': node_features,
            'rel_features_list': rel_features_list,
            'targets_dx': targets_dx,
            'targets_dy': targets_dy,
            'seq_meta': seq_meta,
        }
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info("✓ Cached successfully")
    
    return node_features, rel_features_list, targets_dx, targets_dy, seq_meta

# ============ 运动复杂度估计函数 ============
def compute_motion_complexity(target_dx, target_dy, mask):
    """
    计算x和y轨迹的运动复杂度
    
    Args:
        target_dx: [B, horizon] 目标x方向位移
        target_dy: [B, horizon] 目标y方向位移
        mask: [B, horizon] 有效帧掩码
    
    Returns:
        complexity: [B] 每个样本的复杂度分数（0-1之间，值越大越复杂）
    """
    # 还原为物理单位以便计算
    target_dx = target_dx / 10.0
    target_dy = target_dy / 10.0

    B, H = target_dx.shape
    device = target_dx.device
    
    # 1. 累积位移（轨迹总长度）- 复杂度指标1
    # 计算每一步的位移大小
    step_displacement = torch.sqrt(target_dx**2 + target_dy**2 + 1e-8)  # [B, H]
    # 累积轨迹长度（只考虑有效帧）
    total_path_length = (step_displacement * mask).sum(dim=1)  # [B]
    
    # 2. 方向变化频率和幅度 - 复杂度指标2
    # 计算每一步的方向角度
    angles = torch.atan2(target_dy, target_dx + 1e-8)  # [B, H]
    # 计算方向变化（相邻帧的角度差）
    angle_diff = angles[:, 1:] - angles[:, :-1]  # [B, H-1]
    # 处理周期性（角度差应该在[-π, π]范围内）
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    angle_change_magnitude = torch.abs(angle_diff)  # [B, H-1]
    # 平均方向变化幅度（只考虑有效帧）
    valid_angle_mask = mask[:, 1:] * mask[:, :-1]  # 相邻两帧都有效
    avg_angle_change = (angle_change_magnitude * valid_angle_mask).sum(dim=1) / (valid_angle_mask.sum(dim=1) + 1e-8)  # [B]
    
    # 3. 速度变化（加速度）- 复杂度指标3
    # 计算速度（位移/时间步，假设每步0.1秒）
    velocity = step_displacement / 0.1  # [B, H]
    # 计算加速度（速度变化）
    velocity_diff = velocity[:, 1:] - velocity[:, :-1]  # [B, H-1]
    accel_magnitude = torch.abs(velocity_diff)  # [B, H-1]
    # 平均加速度变化（只考虑有效帧）
    avg_accel_change = (accel_magnitude * valid_angle_mask).sum(dim=1) / (valid_angle_mask.sum(dim=1) + 1e-8)  # [B]
    
    # 4. 曲率（轨迹弯曲程度）- 复杂度指标4
    # 计算曲率：使用三点法（当前点、前一点、后一点）
    # 对于每个中间点，计算曲率
    if H >= 3:
        # 前向向量
        vec1_x = target_dx[:, 1:-1]  # [B, H-2]
        vec1_y = target_dy[:, 1:-1]
        # 后向向量
        vec2_x = target_dx[:, :-2]
        vec2_y = target_dy[:, :-2]
        # 叉积（衡量弯曲程度）
        cross_product = vec1_x * vec2_y - vec1_y * vec2_x
        # 点积（向量长度）
        dot_product = vec1_x * vec2_x + vec1_y * vec2_y
        # 曲率近似（叉积/点积的绝对值）
        curvature = torch.abs(cross_product) / (torch.abs(dot_product) + 1e-8)  # [B, H-2]
        # 平均曲率（只考虑有效帧）
        valid_curvature_mask = mask[:, 1:-1] * mask[:, :-2] * mask[:, 2:]
        avg_curvature = (curvature * valid_curvature_mask).sum(dim=1) / (valid_curvature_mask.sum(dim=1) + 1e-8)  # [B]
    else:
        avg_curvature = torch.zeros(B, device=device)
    
    # 归一化各项指标到[0, 1]范围（使用经验阈值）
    # 路径长度：假设最大为50码（约45.7米）
    norm_path_length = torch.clamp(total_path_length / 50.0, 0, 1)
    # 角度变化：最大为π（180度）
    norm_angle_change = torch.clamp(avg_angle_change / np.pi, 0, 1)
    # 加速度变化：假设最大为10 m/s²
    norm_accel_change = torch.clamp(avg_accel_change / 10.0, 0, 1)
    # 曲率：归一化到[0, 1]
    norm_curvature = torch.clamp(avg_curvature / 2.0, 0, 1)
    
    # 综合复杂度（加权平均）
    complexity = (
        norm_path_length * 0.3 +
        norm_angle_change * 0.3 +
        norm_accel_change * 0.2 +
        norm_curvature * 0.2
    )  # [B]
    
    return complexity


# ============ 训练函数 ============
def train_model_joint(X_train, ydx_train, ydy_train, X_val, ydx_val, ydy_val, 
                      input_dim, horizon, device, batch_size=256, epochs=200, 
                      patience=30, lr=1e-3, rel_train=None, rel_val=None):
    """联合训练 (x, y) - 使用来自 sttf.py 的 STransformer"""
    model = STransformer(
        input_dim, horizon,
        hidden_dim=HIDDEN_DIM,
        n_spatial_layers=2,
        n_temporal_layers=2,
        n_heads=4,
        dropout=0.15,
        window_size=WINDOW_SIZE,
        max_players=K_NEIGHBORS + 1,  # ego + K neighbors
        predict_single_player=True  # 只预测ego球员
    ).to(device)
    logger.info("  🚀 Using STransformer from sttf.py (adapted)")
    logger.info("  ✅ Motion complexity weighting enabled: samples with complexity > 0.5 get 1.25x loss weight")
    
    criterion = TemporalHuber(delta=0.5, time_decay=0.03, lam_smooth=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # 使用 CosineAnnealingWarmRestarts 替代 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr*0.01
    )
    
    # 构建批次（联合 x, y）
    def build_batches(X, Ydx, Ydy, rel_feats):
        batches = []
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            xs = torch.tensor(np.stack(X[i:end]).astype(np.float32))
            
            # 准备 x 和 y 目标
            ydx, mdx = prepare_targets([Ydx[j] for j in range(i, end)], horizon)
            ydy, mdy = prepare_targets([Ydy[j] for j in range(i, end)], horizon)
            
            # 联合为 [B, horizon, 2]
            y_joint = torch.stack([ydx, ydy], dim=-1)  # [B, horizon, 2]
            mask_joint = mdx  # 假设 x 和 y 的 mask 相同
            
            # 构建相对特征批次
            batch_rel = {}
            for key in rel_feats[0].keys():
                batch_rel[key] = torch.tensor(
                    np.stack([rel_feats[j][key] for j in range(i, end)]).astype(np.float32)
                )
            batches.append((xs, y_joint, mask_joint, batch_rel))
        return batches
    
    tr_batches = build_batches(X_train, ydx_train, ydy_train, rel_train)
    va_batches = build_batches(X_val, ydx_val, ydy_val, rel_val)
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        for bx, by, bm, rel_feats in tr_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            rel_feats = {k: v.to(device) for k, v in rel_feats.items()}
            
            # 模型输出 [B, horizon, 2]
            pred = model(bx, rel_feats, ego_idx=0)
            
            # 提取目标值
            target_dx = by[..., 0]  # [B, horizon]
            target_dy = by[..., 1]  # [B, horizon]
            pred_dx = pred[..., 0]  # [B, horizon]
            pred_dy = pred[..., 1]  # [B, horizon]
            
            # ✅ 新增：计算运动复杂度并确定权重
            complexity = compute_motion_complexity(target_dx, target_dy, bm)  # [B]
            # 对复杂度较大的样本（复杂度 > 0.5）应用1.25倍权重
            complexity_threshold = 0.5
            complexity_weight = torch.where(
                complexity > complexity_threshold,
                torch.ones_like(complexity) * 1.25,
                torch.ones_like(complexity)
            )  # [B]
            
            # 计算每个样本的loss（按样本计算，而不是batch平均）
            # 1. Huber loss for x and y (per sample)
            err_x = pred_dx - target_dx  # [B, horizon]
            err_y = pred_dy - target_dy  # [B, horizon]
            abs_err_x = torch.abs(err_x)
            abs_err_y = torch.abs(err_y)
            
            delta = 0.5
            huber_x = torch.where(
                abs_err_x <= delta,
                0.5 * err_x * err_x,
                delta * (abs_err_x - 0.5 * delta)
            )  # [B, horizon]
            huber_y = torch.where(
                abs_err_y <= delta,
                0.5 * err_y * err_y,
                delta * (abs_err_y - 0.5 * delta)
            )  # [B, horizon]
            
            # 时间衰减权重
            if criterion.time_decay and criterion.time_decay > 0:
                L = pred_dx.size(1)
                t = torch.arange(L, device=pred_dx.device, dtype=pred_dx.dtype)
                w = torch.exp(-criterion.time_decay * t).view(1, L)
                huber_x = huber_x * w
                huber_y = huber_y * w
                bm_weighted = bm * w
            else:
                bm_weighted = bm
            
            # 每个样本的loss（按样本维度求平均）
            loss_x_per_sample = (huber_x * bm_weighted).sum(dim=1) / (bm_weighted.sum(dim=1) + 1e-8)  # [B]
            loss_y_per_sample = (huber_y * bm_weighted).sum(dim=1) / (bm_weighted.sum(dim=1) + 1e-8)  # [B]
            
            # 2. 方向一致性损失（按样本计算）
            pred_angle = torch.atan2(pred_dy, pred_dx + 1e-8)  # [B, horizon]
            target_angle = torch.atan2(target_dy, target_dx + 1e-8)  # [B, horizon]
            angle_diff = torch.abs(pred_angle - target_angle)
            angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)  # 处理周期性
            direction_loss_per_sample = (angle_diff * bm_weighted).sum(dim=1) / (bm_weighted.sum(dim=1) + 1e-8) * 0.1  # [B]
            
            # 3. 平滑损失（按样本计算）
            smooth_loss_per_sample = torch.zeros_like(loss_x_per_sample)  # [B]
            if criterion.lam_smooth and pred_dx.size(1) > 2:
                d1_x = pred_dx[:, 1:] - pred_dx[:, :-1]
                d2_x = d1_x[:, 1:] - d1_x[:, :-1]
                d1_y = pred_dy[:, 1:] - pred_dy[:, :-1]
                d2_y = d1_y[:, 1:] - d1_y[:, :-1]
                m2 = bm_weighted[:, 2:]
                smooth_x = (d2_x * d2_x) * m2
                smooth_y = (d2_y * d2_y) * m2
                smooth_loss_per_sample = (
                    (smooth_x.sum(dim=1) / (m2.sum(dim=1) + 1e-8)) +
                    (smooth_y.sum(dim=1) / (m2.sum(dim=1) + 1e-8))
                ) / 2.0 * criterion.lam_smooth  # [B]
            
            # 每个样本的总loss
            loss_per_sample = (loss_x_per_sample + loss_y_per_sample) / 2.0 + direction_loss_per_sample + smooth_loss_per_sample  # [B]
            
            # ✅ 应用复杂度权重：对高复杂度样本的loss乘以1.25
            loss_per_sample_weighted = loss_per_sample * complexity_weight  # [B]
            
            # Batch平均（加权后的loss）
            loss_weighted = loss_per_sample_weighted.mean()
            
            # 记录原始loss（用于日志）
            loss = loss_per_sample.mean()
            
            optimizer.zero_grad()
            loss_weighted.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())  # 记录原始loss用于日志
        
        scheduler.step()  # CosineAnnealing 每个 epoch 更新
        
        # 记录复杂度统计（每10个epoch）
        if epoch % 10 == 0:
            # 计算一个batch的复杂度统计（用于日志）
            if len(tr_batches) > 0:
                bx_sample, by_sample, bm_sample, rel_feats_sample = tr_batches[0]
                bx_sample = bx_sample.to(device)
                by_sample = by_sample.to(device)
                bm_sample = bm_sample.to(device)
                rel_feats_sample = {k: v.to(device) for k, v in rel_feats_sample.items()}
                with torch.no_grad():
                    pred_sample = model(bx_sample, rel_feats_sample, ego_idx=0)
                    target_dx_sample = by_sample[..., 0]
                    target_dy_sample = by_sample[..., 1]
                    complexity_sample = compute_motion_complexity(target_dx_sample, target_dy_sample, bm_sample)
                    high_complexity_ratio = (complexity_sample > 0.5).float().mean().item()
                    avg_complexity = complexity_sample.mean().item()
                    logger.info(f"  [Complexity Stats] avg={avg_complexity:.3f}, high_complexity_ratio={high_complexity_ratio:.2%}")
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm, rel_feats in va_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                rel_feats = {k: v.to(device) for k, v in rel_feats.items()}
                pred = model(bx, rel_feats, ego_idx=0)
                loss_x = criterion(pred[..., 0], by[..., 0], bm)
                loss_y = criterion(pred[..., 1], by[..., 1], bm)
                loss = (loss_x + loss_y) / 2.0
                val_losses.append(loss.item())
        
        trl, val = float(np.mean(train_losses)), float(np.mean(val_losses))
        
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}: train={trl:.4f}, val={val:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        if val < best_loss:
            best_loss, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss


def compute_val_rmse_joint(model, X_val_sc, ydx_list, ydy_list, horizon, device, rel_val):
    """计算验证集RMSE - 联合预测模型"""
    X_t = torch.tensor(X_val_sc.astype(np.float32)).to(device)
    with torch.no_grad():
        # 构建相对特征
        rel_feats = {}
        for key in rel_val[0].keys():
            rel_feats[key] = torch.tensor(
                np.stack([rel_val[i][key] for i in range(len(rel_val))]).astype(np.float32)
            ).to(device)
        
        # 模型输出 [B, horizon, 2]
        pred_joint = model(X_t, rel_feats, ego_idx=0).cpu().numpy()
        
        # Unscale and Cumsum to get positions
        pdx_vel = pred_joint[..., 0] / 10.0
        pdy_vel = pred_joint[..., 1] / 10.0
        pdx = np.cumsum(pdx_vel, axis=1)
        pdy = np.cumsum(pdy_vel, axis=1)
    
    ydx, m = prepare_targets(ydx_list, horizon)
    ydy, _ = prepare_targets(ydy_list, horizon)
    ydx, ydy, m = ydx.numpy(), ydy.numpy(), m.numpy()
    
    # Unscale and Cumsum targets to get positions
    ydx = np.cumsum(ydx / 10.0, axis=1)
    ydy = np.cumsum(ydy / 10.0, axis=1)
    
    se_sum2d = ((pdx - ydx)**2 + (pdy - ydy)**2) * m
    denom = m.sum() + 1e-8
    
    # per-dim RMSE
    return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))


# ============ 主训练流程 ============
def main_train():
    import gc
    
    logger.info("="*80)
    logger.info("RNN TRAINING MODE (with Direction Unification)")
    logger.info("="*80)
    
    # 1. 加载数据
    logger.info("[1/5] Loading training data...")
    input_data, output_data = load_all_train_data()
    
    # 2. 方向统一处理
    logger.info("[2/5] Direction Unification...")
    logger.info("Building play direction map...")
    direction_map = build_play_direction_map(input_data)
    logger.info(f"Direction distribution: {input_data.select('play_direction').to_pandas()['play_direction'].value_counts().to_dict()}")
    
    # 给 output_data 补上方向列（通过 join）
    logger.info("Adding play_direction to output_data...")
    dir_df = input_data.select(['game_id', 'play_id', 'play_direction']).unique()
    output_data = output_data.join(dir_df, on=['game_id', 'play_id'], how='left')
    
    logger.info("Unifying all plays to 'left' direction...")
    input_data = unify_left_direction(input_data)
    output_data = unify_left_direction(output_data)
    logger.info("✓ Direction unified to 'left'")
    
    # 3. 特征工程 (使用 besttmp.py 的特征)
    logger.info("[3/5] Feature Engineering...")
    logger.info("Step 1: Engineering advanced physics features...")
    input_features = engineer_advanced_features(input_data)
    del input_data  # 释放原始数据
    gc.collect()
    
    logger.info("Step 2: Adding heading features...")
    input_features = add_heading_features(input_features)
    gc.collect()
    
    logger.info("Step 3: Adding sequence and rolling features...")
    input_features = add_sequence_features(input_features)
    gc.collect()
    
    logger.info("Step 4: Adding trajectory and prediction features...")
    input_features = add_trajectory_and_prediction_features(input_features)
    gc.collect()
    
    logger.info("Step 4b: Adding physics constraint features...")
    input_features = add_physics_constraint_features(input_features)
    gc.collect()
    
    logger.info("Step 5: Computing per-frame GNN neighbor embeddings (time-varying spatial features)...")
    gnn_features = compute_neighbor_embeddings_per_frame(input_features, k_neigh=6, radius=30.0, tau=8.0)
    gc.collect()
    
    logger.info("Step 5b: Computing advanced graph features...")
    advanced_graph_features = compute_advanced_graph_features(input_features, k_neigh=6, radius=30.0)
    gc.collect()
    
    logger.info("Step 6: Merging per-frame GNN neighbor embeddings...")
    input_features = input_features.join(gnn_features, on=['game_id', 'play_id', 'frame_id', 'nfl_id'], how='left')
    del gnn_features  # 释放GNN特征
    gc.collect()
    
    logger.info("Step 6b: Merging advanced graph features...")
    input_features = input_features.join(advanced_graph_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    del advanced_graph_features  # 释放高级图特征
    gc.collect()
    
    logger.info("Step 6c: Adding football rule features...")
    input_features = add_football_rule_features(input_features)
    gc.collect()
    
    logger.info("Step 7: Adding time features...")
    input_features = add_time_features(input_features)
    gc.collect()
    
    logger.info("Step 8: Adding QB relative features...")
    input_features = add_qb_relative_features(input_features)
    gc.collect()
    
    logger.info("Step 9: Cleaning features (removing inf/nan/extreme values)...")
    input_features = clean_features_for_modeling(input_features)
    gc.collect()
    
    logger.info(f"Feature engineered data shape: {input_features.shape}")
    logger.info(f"Total features: {len(input_features.columns)}")
    
    # 3. 定义特征列
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir',
        'velocity_x', 'velocity_y', 'dist_to_ball', 'angle_to_ball',
        'velocity_toward_ball', 'time_to_ball', 'orientation_diff',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer',
        'side_offense', 'height_inches', 'player_weight', 'bmi',
        'ball_land_x', 'ball_land_y', 'num_frames_output', 'frame_id',
        'acceleration_x', 'acceleration_y', 'distance_to_target_x', 'distance_to_target_y',
        'speed_squared', 'accel_magnitude', 'velocity_alignment',
        'expected_x_at_ball', 'expected_y_at_ball',
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'angle_diff', 'time_squared', 'dist_squared', 'weighted_dist_by_time',
        'heading_x', 'heading_y',
        # GNN features
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
        # Advanced graph features
        'graph_hop1_neighbors', 'graph_extended_neighbors', 'graph_extended_avg_dist',
        'graph_neighbor_avg_degree', 'graph_spatial_cv',
        'local_space_control', 'min_neighbor_dist',
        'tactical_dist_to_passer', 'tactical_dist_to_target', 'tactical_defenders_nearby',
        'pressure_total', 'pressure_max_threat', 'pressure_close_opponents',
        # Spatial features
        'field_position_x', 'field_position_y', 'field_center_distance',
        'in_endzone', 'in_midfield', 'in_redzone', 'field_side',
        'distance_to_sideline', 'distance_to_goal_line', 'distance_to_endzone',
        'near_sideline', 'near_goal_line', 'near_center',
        'field_quadrant', 'play_direction_right', 'relative_x_position',
        'yardline_position', 'yardline_normalized', 'distance_to_goal',
        'in_pressure_zone',
        # Trajectory features
        'predicted_x_linear', 'predicted_y_linear', 'predicted_x_accel', 'predicted_y_accel',
        'prediction_error_x', 'prediction_error_y', 'prediction_error_total',
        'prediction_error_rolling_mean', 'velocity_change_rate',
        'dir_change', 'dir_change_abs', 'trajectory_curvature', 'trajectory_curvature_abs',
        'curvature_rolling_mean', 'curvature_rolling_std', 'sharp_turn',
        # Interception features
        'interception_speed_needed', 'speed_ratio_to_intercept',
        'velocity_alignment_to_ball', 'interception_feasibility',
        'interception_prob_weighted', 'can_intercept_in_time',
        'intercept_angle_deviation', 'interception_efficiency',
        'relative_dist_to_ball', 'relative_speed_to_ball', 'interception_advantage',
        # Football rule features
        'defenders_at_catch_point', 'receivers_at_catch_point',
        'catch_point_advantage', 'is_contested_catch',
        'in_redzone_pass', 'near_goalline_pass', 'two_point_zone',
        'yards_to_midfield', 'redzone_congestion', 'endzone_vertical_constraint',
        # Time features
        'frames_elapsed', 'normalized_time', 'time_sin', 'time_cos',
        # QB relative features
        'qb_distance', 'vel_to_qb_alignment', 'vel_to_qb_perp',
        'bearing_to_qb_signed', 'bearing_to_qb_sin', 'bearing_to_qb_cos',
        # Physics constraint features
        'displacement_error_total', 'displacement_consistency_score',
        'actual_motion_dir', 'direction_deviation', 'direction_deviation_abs',
    ]
    
    # 添加滞后和滚动特征
    for lag in [1, 2, 3, 4, 5, 7]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    
    for window in [3, 5, 7, 10]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.extend([
                f'{col}_rolling_mean_{window}',
                f'{col}_rolling_std_{window}',
            ])
    
    for col in ['s', 'velocity_x', 'velocity_y']:
        feature_cols.extend([
            f'{col}_diff1', f'{col}_diff2',
            f'{col}_trend_3', f'{col}_trend_5', f'{col}_trend_10',
            f'{col}_trend_accel'
        ])
    
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    
    # 过滤可用特征
    available_features = [col for col in feature_cols if col in input_features.columns]
    logger.info(f"Available features: {len(available_features)}")
    
    # 4. 创建序列 (支持缓存)
    logger.info("[4/5] Creating sequences...")
    train_cache_path = None
    if USE_CACHE:
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        train_cache_path = cache_dir / "train_sequences.pkl"
        logger.info(f"Cache enabled: {train_cache_path}")
    else:
        logger.info("Cache disabled (--no_cache flag)")
    
    # 使用时空子图格式
    logger.info("📊 Using spatiotemporal subgraphs (ego + K neighbors)")
    subgraph_cache = cache_dir / "subgraphs.pkl" if USE_CACHE else None
    sequences, rel_features_list, targets_dx, targets_dy, seq_meta = create_spatiotemporal_subgraphs(
        input_features, output_data, available_features,
        window_size=WINDOW_SIZE,
        k_neighbors=K_NEIGHBORS,
        cache_path=subgraph_cache
    )
    
    # 检查序列是否为空
    if len(sequences) == 0:
        logger.error("ERROR: No sequences created! Check data and feature engineering.")
        logger.error(f"Input features shape: {input_features.shape}")
        logger.error(f"Output data shape: {output_data.shape}")
        logger.error(f"Available features count: {len(available_features)}")
        raise ValueError("No sequences created - cannot proceed with training!")
    
    logger.info(f"✓ Created {len(sequences)} sequences successfully")
    logger.info(f"  Sequence shape: {sequences[0].shape}")
    logger.info(f"  Feature dimension: {len(available_features)}")
    
    # 🗑️ 释放内存：删除不再需要的大型 DataFrame
    logger.info("Releasing memory: cleaning up intermediate data...")
    del input_features, output_data
    # 清理可能存在的其他中间变量
    if 'dir_df' in locals():
        del dir_df
    if 'direction_map' in locals():
        del direction_map
    gc.collect()
    logger.info("✓ Memory cleaned")
    
    # 5. 多种子 K 折训练
    logger.info("[5/5] Multi-seed K-Fold training...")
    
    # 创建保存目录
    save_dir = Path(MODEL_SAVE_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存元信息
    meta = {
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "feature_cols": available_features,
        "window_size": WINDOW_SIZE,
        "max_future_horizon": MAX_FUTURE_HORIZON,
        "hidden_dim": HIDDEN_DIM,
        "use_spatiotemporal": USE_SPATIOTEMPORAL,
        "k_neighbors": K_NEIGHBORS,
        "version": 5,  # Enhanced STransformer with joint (x,y) prediction
        "model_type": "joint",  # 标记为联合预测模型
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"✓ Meta info saved to {save_dir / 'meta.json'}")
    
    # 准备分组
    groups = np.array([f"{d['game_id']}_{d['play_id']}" for d in seq_meta])
    
    all_rmse = []
    cv_log = []
    
    for seed in SEEDS:
        logger.info("="*70)
        logger.info(f"Seed {seed}")
        logger.info("="*70)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        gkf = GroupKFold(n_splits=N_FOLDS)
        seed_dir = save_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        
        fold_rmses = []
        
        for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
            logger.info("-"*60)
            logger.info(f"Fold {fold}/{N_FOLDS} (seed {seed})")
            logger.info("-"*60)
            
            X_tr = [sequences[i] for i in tr]
            X_va = [sequences[i] for i in va]
            
            # 准备相对特征
            rel_tr = [rel_features_list[i] for i in tr]
            rel_va = [rel_features_list[i] for i in va]
            
            # 标准化: [T, N, D] -> 重塑为 [T*N, D] 做标准化
            scaler = StandardScaler()
            X_tr_flat = [s.reshape(-1, s.shape[-1]) for s in X_tr]
            scaler.fit(np.vstack(X_tr_flat))
            X_tr_sc = np.stack([
                scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
                for s in X_tr
            ]).astype(np.float32)
            X_va_sc = np.stack([
                scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
                for s in X_va
            ]).astype(np.float32)
            
            # 释放原始序列数据
            del X_tr, X_va
            gc.collect()
            
            # 联合训练 (X, Y) 模型
            logger.info("Training joint (ΔX, ΔY) model...")
            model, loss = train_model_joint(
                X_tr_sc, 
                [targets_dx[i] for i in tr],
                [targets_dy[i] for i in tr],
                X_va_sc,
                [targets_dx[i] for i in va],
                [targets_dy[i] for i in va],
                X_tr_sc.shape[-1], MAX_FUTURE_HORIZON, DEVICE,
                batch_size=BATCH_SIZE, epochs=EPOCHS, patience=PATIENCE, lr=LEARNING_RATE,
                rel_train=rel_tr,
                rel_val=rel_va
            )
            
            # 计算验证RMSE
            rmse = compute_val_rmse_joint(
                model, X_va_sc,
                [targets_dx[i] for i in va],
                [targets_dy[i] for i in va],
                MAX_FUTURE_HORIZON, DEVICE,
                rel_val=rel_va
            )
            
            logger.info(f"[VAL] seed {seed} fold {fold} → "
                  f"Joint Loss={loss:.5f} | RMSE={rmse:.4f} yards")
            
            fold_rmses.append(rmse)
            all_rmse.append(rmse)
            cv_log.append({
                "seed": seed, "fold": fold,
                "rmse": rmse,
                "loss_joint": float(loss),
            })
            
            # 保存模型（现在只保存一个联合模型）
            joblib.dump(scaler, seed_dir / f"scaler_fold{fold}.pkl")
            torch.save(model.state_dict(), seed_dir / f"model_joint_fold{fold}.pt")
            logger.info(f"✓ Joint model saved for seed {seed} fold {fold}")
            
            # 释放模型和数据
            del model, scaler, X_tr_sc, X_va_sc, rel_tr, rel_va
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
              f"mean={float(np.mean(fold_rmses)):.4f} yards")
    
    # 最终汇总
    logger.info("="*80)
    logger.info(f"[CV SUMMARY] all folds RMSEs: {[f'{r:.4f}' for r in all_rmse]}")
    logger.info(f"[CV SUMMARY] overall mean RMSE = {float(np.mean(all_rmse)):.4f} yards")
    logger.info("="*80)
    
    # 保存CV指标
    with open(save_dir / "cv_metrics.json", "w") as f:
        json.dump({"per_fold": cv_log, "overall_mean": float(np.mean(all_rmse))}, f, indent=2)
    logger.info(f"✓ CV metrics saved to {save_dir / 'cv_metrics.json'}")
    
    logger.info(f"✓ Training complete! Models saved to: {save_dir}")
    logger.info(f"Total models: {len(SEEDS) * N_FOLDS} per coordinate")


if MODE == 'infer':
    # 推理模式：移除控制台处理器，只保留文件日志
    logger_obj = logging.getLogger()
    console_handlers = [h for h in logger_obj.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    for handler in console_handlers:
        logger_obj.removeHandler(handler)
    
    logger.info("Loading saved models and meta info...")
    save_dir = "/kaggle/input/nfl-gru"
    save_dir = Path(save_dir)
    meta_path = save_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    saved_feature_cols = meta["feature_cols"]
    saved_window = int(meta.get("window_size", WINDOW_SIZE))
    horizon = int(meta.get("max_future_horizon", MAX_FUTURE_HORIZON))
    hidden_dim = int(meta.get("hidden_dim", HIDDEN_DIM))
    saved_k_neighbors = int(meta.get("k_neighbors", K_NEIGHBORS))
    seeds = meta["seeds"]
    n_folds = int(meta["n_folds"])
    
    logger.info(f"✓ Loaded meta: {len(saved_feature_cols)} features, window={saved_window}, horizon={horizon}")
    logger.info(f"  Enhanced STransformer with K={saved_k_neighbors} neighbors (joint prediction)")
    
    models, scalers = [], []
    for seed in seeds:
        seed_dir = save_dir / f"seed_{seed}"
        for fold in range(1, n_folds + 1):
            sc_path = seed_dir / f"scaler_fold{fold}.pkl"
            model_path = seed_dir / f"model_joint_fold{fold}.pt"
            
            if not (sc_path.exists() and model_path.exists()):
                logger.warning(f"Missing seed={seed} fold={fold}, skip")
                continue
            
            scaler = joblib.load(sc_path)
            
            # STransformer from sttf.py (adapted)
            model = STransformer(
                len(saved_feature_cols), horizon,
                hidden_dim=hidden_dim,
                n_spatial_layers=2,
                n_temporal_layers=2,
                n_heads=4,
                dropout=0.15,
                window_size=saved_window,
                max_players=saved_k_neighbors + 1,  # ego + K neighbors
                predict_single_player=True
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            scalers.append(scaler)
            models.append(model)
    
    logger.info(f"✓ Loaded {len(models)} joint (x,y) models")

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """推理函数 - 使用 STransformer"""
    import gc    
    # 方向统一处理
    test_input = unify_left_direction(test_input)
    
    # 特征工程
    test_features = engineer_advanced_features(test_input)
    del test_input
    gc.collect()
    
    test_features = add_heading_features(test_features)
    test_features = add_sequence_features(test_features)
    test_features = add_trajectory_and_prediction_features(test_features)
    test_features = add_physics_constraint_features(test_features)
    
    gnn_test = compute_neighbor_embeddings_per_frame(test_features, k_neigh=6, radius=30.0, tau=8.0)
    test_features = test_features.join(gnn_test, on=['game_id', 'play_id', 'frame_id', 'nfl_id'], how='left')
    del gnn_test
    gc.collect()
    
    advanced_graph_test = compute_advanced_graph_features(test_features, k_neigh=6, radius=30.0)
    test_features = test_features.join(advanced_graph_test, on=['game_id', 'play_id', 'nfl_id'], how='left')
    del advanced_graph_test
    gc.collect()
    
    test_features = add_football_rule_features(test_features)
    test_features = add_time_features(test_features)
    test_features = add_qb_relative_features(test_features)
    test_features = clean_features_for_modeling(test_features)
    
    # 检查特征一致性
    for col in saved_feature_cols:
        if col not in test_features.columns:
            test_features = test_features.with_columns([pl.lit(0.0).alias(col)])
    
    # 创建时空子图（伪造 output 模板）
    test_template_pl = test.clone()
    test_seqs, test_rel_feats, _, _, test_meta = create_spatiotemporal_subgraphs(
        test_features, test_template_pl, saved_feature_cols,
        window_size=saved_window,
        k_neighbors=saved_k_neighbors,
        cache_path=None,
        show_progress=False  # 推理时不显示进度条
    )
    
    del test_features
    gc.collect()
    
    # 集成预测（联合模型）
    all_preds_joint = []
    
    for scaler, model in zip(scalers, models):
        # 标准化
        X_test_sc = np.stack([
            scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
            for s in test_seqs
        ]).astype(np.float32)
        
        # 构建相对特征
        rel_feats = {}
        for key in test_rel_feats[0].keys():
            rel_feats[key] = torch.tensor(
                np.stack([test_rel_feats[i][key] for i in range(len(test_rel_feats))]).astype(np.float32)
            ).to(DEVICE)
        
        X_t = torch.tensor(X_test_sc).to(DEVICE)
        
        with torch.no_grad():
            pred_joint = model(X_t, rel_feats, ego_idx=0).cpu().numpy()  # [B, horizon, 2]
        
        all_preds_joint.append(pred_joint)
    
    # 集成平均
    ens_joint = np.mean(all_preds_joint, axis=0)  # [B, horizon, 2]
    
    # Unscale and Cumsum to get displacement from last_frame
    ens_vel_x = ens_joint[..., 0] / 10.0
    ens_vel_y = ens_joint[..., 1] / 10.0
    ens_dx = np.cumsum(ens_vel_x, axis=1)  # [B, horizon]
    ens_dy = np.cumsum(ens_vel_y, axis=1)  # [B, horizon]
    
    H = ens_dx.shape[1]
    idx_x = saved_feature_cols.index('x')
    idx_y = saved_feature_cols.index('y')
    
    test_pd = test.to_pandas()
    test_idx = test_pd.set_index(['game_id', 'play_id', 'nfl_id']).sort_index()
    
    rows = []
    for i, meta_row in enumerate(test_meta):
        gid = meta_row['game_id']
        pid = meta_row['play_id']
        nid = meta_row['nfl_id']
        play_dir = meta_row.get('play_direction', None)
        play_is_right = (play_dir == 'right')
        
        # 最后一帧ego的位置
        last_x = test_seqs[i][-1, 0, idx_x]
        last_y = test_seqs[i][-1, 0, idx_y]
        
        try:
            fids = test_idx.loc[(gid, pid, nid), 'frame_id']
            if isinstance(fids, pd.Series):
                fids = fids.sort_values().tolist()
            else:
                fids = [int(fids)]
        except KeyError:
            continue
        
        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            # 统一方向的预测坐标
            x_uni = np.clip(last_x + ens_dx[i, tt], 0, FIELD_LENGTH)
            y_uni = np.clip(last_y + ens_dy[i, tt], 0, FIELD_WIDTH)
            
            # 转换回原始方向
            x_pred, y_pred = invert_to_original_direction(x_uni, y_uni, play_is_right)
            
            rows.append({
                'x': x_pred,
                'y': y_pred
            })
    
    predictions = pd.DataFrame(rows)
    return predictions


if __name__ == "__main__":
    if MODE == 'train':
        main_train()
    elif MODE == 'infer':
        import kaggle_evaluation.nfl_inference_server
        # Initialize inference server
        inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)
        
        # Start server in competition environment
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            logger.info("[SERVER] Starting inference server...")
            inference_server.serve()
        else:
            logger.info("[SERVER] Running local gateway for testing...")
            inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
    else:
        raise ValueError(f"Invalid MODE: {MODE}. Use 'train' or 'infer'")