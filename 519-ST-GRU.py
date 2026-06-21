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
parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer'], 
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
N_FOLDS = 5
SEEDS = [42, 3507, 2025, 114514, 2026, 2027]
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

def transform_to_relative_coordinates(df):
    """
    Transform coordinates to be relative to ball landing point (origin = ball_land).
    Preserves absolute coordinates in x_abs, y_abs for field-based features.
    Sets ball_land_x/y columns to 0 to maintain consistency in feature formulas.
    """
    # 1. Save absolute coordinates if they don't exist (or overwrite if needed)
    # We prioritize preserving 'x' as 'x_abs' before modifying 'x'
    df = df.with_columns([
        pl.col('x').alias('x_abs'),
        pl.col('y').alias('y_abs'),
        pl.col('ball_land_x').alias('ball_land_x_abs'),
        pl.col('ball_land_y').alias('ball_land_y_abs')
    ])
    
    # 2. Shift x and y to relative coordinates
    df = df.with_columns([
        (pl.col('x') - pl.col('ball_land_x')).alias('x'),
        (pl.col('y') - pl.col('ball_land_y')).alias('y')
    ])
    
    # 3. Zero out ball_land_x/y so downstream features (like dist_to_ball) 
    # that calculate (x - ball_land_x) still work correctly (now: x_rel - 0 = x_rel)
    df = df.with_columns([
        pl.lit(0.0).alias('ball_land_x'),
        pl.lit(0.0).alias('ball_land_y')
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
    # Determine which columns to use for field position (absolute coordinates)
    # If x_abs exists (from transform_to_relative_coordinates), use it. Otherwise use x.
    x_field = pl.col('x_abs') if 'x_abs' in df.columns else pl.col('x')
    y_field = pl.col('y_abs') if 'y_abs' in df.columns else pl.col('y')

    df = df.with_columns([
        (x_field / 120.0).alias('field_position_x'),
        (y_field / 53.3).alias('field_position_y'),
        ((x_field - 60)**2 + (y_field - 26.65)**2).sqrt().alias('field_center_distance'),
    ])
    
    df = df.with_columns([
        ((x_field < 10) | (x_field > 110)).cast(pl.Int32).alias('in_endzone'),
        ((x_field >= 50) & (x_field <= 70)).cast(pl.Int32).alias('in_midfield'),
        (((x_field >= 0) & (x_field <= 20)) | ((x_field >= 100) & (x_field <= 120))).cast(pl.Int32).alias('in_redzone'),
        pl.when(x_field <= 60).then(0).otherwise(1).alias('field_side'),
    ])
    
    df = df.with_columns([
        pl.min_horizontal([y_field, 53.3 - y_field]).alias('distance_to_sideline'),
        pl.min_horizontal([x_field, 120 - x_field]).alias('distance_to_goal_line'),
        pl.min_horizontal([x_field, 120 - x_field]).alias('distance_to_endzone'),
    ])
    
    df = df.with_columns([
        (pl.col('distance_to_sideline') < 5).cast(pl.Int32).alias('near_sideline'),
        (pl.col('distance_to_goal_line') < 10).cast(pl.Int32).alias('near_goal_line'),
        (pl.col('field_center_distance') < 10).cast(pl.Int32).alias('near_center'),
    ])
    
    # Field quadrant
    df = df.with_columns([
        pl.when((x_field <= 60) & (y_field <= 26.65))
            .then(1)
            .when((x_field > 60) & (y_field <= 26.65))
            .then(2)
            .when((x_field <= 60) & (y_field > 26.65))
            .then(3)
            .otherwise(4)
            .alias('field_quadrant'),
    ])
    
    df = df.with_columns([
        (pl.col('play_direction') == 'right').cast(pl.Int32).alias('play_direction_right'),
    ])
    
    df = df.with_columns([
        pl.when(pl.col('play_direction_right') == 1)
            .then(x_field)
            .otherwise(120 - x_field)
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


# ============ 空间特征增强模块 ============
class FourierFeatureEncoder(nn.Module):
    """多频率傅里叶编码 - 优化张量操作"""
    def __init__(self, bands=(1, 2, 4, 8, 16)):
        super().__init__()
        self.bands = bands
        # 预计算2π避免重复计算
        self.register_buffer('two_pi', torch.tensor(2 * np.pi))
        
    def forward(self, x):
        """
        x: [..., C] 输入特征 (如 x, y 坐标)
        输出: [..., C*(1+2*len(bands))] 扩展特征
        """
        outputs = [x]
        # 一次性计算所有频带，减少重复操作
        for band in self.bands:
            band_tensor = torch.tensor(band, device=x.device, dtype=x.dtype)
            angle = self.two_pi * band_tensor * x
            outputs.append(torch.sin(angle))
            outputs.append(torch.cos(angle))
        return torch.cat(outputs, dim=-1)


class RBFEncoder(nn.Module):
    """径向基函数编码 - 优化计算"""
    def __init__(self, num_centers=10, min_val=0.0, max_val=30.0):
        super().__init__()
        centers = torch.linspace(min_val, max_val, num_centers)
        self.register_buffer('centers', centers)
        self.sigma = (max_val - min_val) / (num_centers - 1)
        # 预计算sigma平方避免重复计算
        self.register_buffer('sigma_sq', torch.tensor(self.sigma ** 2))
        
    def forward(self, x):
        """
        x: [..., 1] 距离特征
        输出: [..., num_centers] RBF 编码
        """
        # 使用broadcasting避免显式reshape
        diff = x.unsqueeze(-1) - self.centers  # [..., num_centers]
        return torch.exp(-0.5 * (diff ** 2) / self.sigma_sq)

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

class ResidualMLP(nn.Module):
    """MLP - 优化内存布局，输出 [B, horizon, 2] (x, y联合预测)"""
    def __init__(self, d_in, d_hidden, horizon, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        # 分别预测 x 和 y，但共享特征提取
        self.out_x = nn.Linear(d_hidden, horizon)
        self.out_y = nn.Linear(d_hidden, horizon)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        # 保持内存连续性，删除残差连接
        y = self.drop(self.act(self.fc1(x)))
        y = self.drop(self.act(self.fc2(y)))
        
        # 分别预测 x 和 y 维度
        pred_x = self.out_x(y)  # [B, horizon]
        pred_y = self.out_y(y)  # [B, horizon]
        
        # 拼接为 [B, horizon, 2]
        return torch.stack([pred_x, pred_y], dim=-1)

class RelativePositionBias(nn.Module):
    """相对位置偏置 - 优化特征拼接"""
    def __init__(self, n_heads=4):
        super().__init__()
        # 输入维度计算: Fourier(2) + dist + angle + dv + ally = 2*(1+2*4) + 1 + 1 + 2 + 1 = 23
        input_dim = 2 * (1 + 2 * len((1, 2, 4, 8))) + 1 + 1 + 2 + 1
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_heads)
        )
        self.fourier = FourierFeatureEncoder(bands=(1, 2, 4, 8))
        
    def forward(self, rel_features):
        """
        rel_features: dict with keys 'dxy', 'dist', 'angle', 'dv', 'ally_mask'
        return: [B, T, n_heads, N, N] 加性偏置
        """
        B, T, N, _, _ = rel_features['dxy'].shape
        
        # 优化：展平批处理和时间维度，减少重复计算
        dxy_flat = rel_features['dxy'].reshape(B*T*N*N, 2)
        dxy_fourier = self.fourier(dxy_flat).reshape(B, T, N, N, -1)
        
        # 特征拼接优化：避免多次索引
        features = torch.cat([
            dxy_fourier,
            rel_features['dist'],
            rel_features['angle'],
            rel_features['dv'],
            rel_features['ally_mask']
        ], dim=-1)
        
        bias = self.proj(features)
        return bias.permute(0, 1, 4, 2, 3)

class FrameSpatialAttention(nn.Module):
    """帧内空间注意力 - 优化掩码和计算"""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # 使用nn.Linear的bias=False减少参数
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rel_bias = RelativePositionBias(n_heads=n_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # 预计算缩放因子
        self.register_buffer('scale', torch.tensor(self.head_dim ** -0.5))
        
    def forward(self, x, rel_features, mask=None):
        B, T, N, D = x.shape
        
        # 优化：保持内存连续性，避免多次reshape
        q = self.q_proj(x).view(B, T, N, self.n_heads, self.head_dim).transpose(2, 3)
        k = self.k_proj(x).view(B, T, N, self.n_heads, self.head_dim).transpose(2, 3)
        v = self.v_proj(x).view(B, T, N, self.n_heads, self.head_dim).transpose(2, 3)
        
        # 优化：使用einsum提高可读性和性能
        scores = torch.einsum('bthnd,bthmd->bthnm', q, k) * self.scale
        
        # 添加相对位置偏置
        bias = self.rel_bias(rel_features)
        scores = scores + bias
        
        # 优化掩码处理：避免多次类型转换
        if mask is not None:
            # mask: [B, T, N] ->扩展到注意力分数形状
            attn_mask = mask.unsqueeze(2).unsqueeze(3).bool()
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        
        # 优化：使用softmax的dtype参数提高效率
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)
        
        # 优化：使用einsum
        out = torch.einsum('bthnm,bthmd->bthnd', attn, v)
        out = out.transpose(2, 3).reshape(B, T, N, D)
        
        out = self.out_proj(out)
        return self.layer_norm(x + out)

class SpatioTemporal_GRU(nn.Module):
    """SpatioTemporal_GRU - 优化RNN和内存使用，联合(x,y)预测"""
    def __init__(self, input_dim, horizon, hidden_dim=128, n_spatial_layers=2, 
                 n_temporal_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_temporal_layers = n_temporal_layers
        self.horizon = horizon
        # 输入投影 - 简化版本
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 帧内空间注意力层
        self.spatial_layers = nn.ModuleList([
            FrameSpatialAttention(hidden_dim, n_heads, dropout)
            for _ in range(n_spatial_layers)
        ])
        
        # 帧间时间编码 - 使用更高效的GRU实现（单向）
        self.temporal_gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_temporal_layers,
            batch_first=True, dropout=dropout if n_temporal_layers > 1 else 0
        )
        
        # 预测头
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        # 使用小初始化避免初始预测过大
        self.pool_query = nn.Parameter(torch.randn(1, 2, hidden_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        
        # 分段预测头（short/mid/long）用于联合预测
        short_len = min(15, horizon)
        remaining = max(horizon - short_len, 0)
        mid_len = min(15, remaining)
        long_len = max(remaining - mid_len, 0)
        self.segment_lengths = (short_len, mid_len, long_len)
        # 每个分段输出 [B, segment_len, 2]
        self.head_short = ResidualMLP(hidden_dim * 2, hidden_dim, short_len) if short_len > 0 else None
        self.head_mid = ResidualMLP(hidden_dim * 2, hidden_dim, mid_len) if mid_len > 0 else None
        self.head_long = ResidualMLP(hidden_dim * 2, hidden_dim, long_len) if long_len > 0 else None
        
    def forward(self, x, rel_features, ego_idx=0, mask=None):
        B, T, N, D = x.shape
        
        # 输入投影 - 保持内存连续
        h = self.input_proj(x).contiguous()
        
        # 帧内空间注意力 - 使用in-place操作减少内存
        for spatial_layer in self.spatial_layers:
            h = spatial_layer(h, rel_features, mask)
        
        # 提取 ego 节点的时序特征
        ego_features = h[:, :, ego_idx, :]  # [B, T, hidden_dim]
        
        # 时间编码 - 优化：h0初始化为None让GRU自动处理
        temporal_out, _ = self.temporal_gru(ego_features)
        
        # 注意力池化 - 优化：保持批处理维度一致
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(temporal_out), self.pool_ln(temporal_out))
        ctx = ctx.reshape(B, -1)  # [B, hidden_dim*2]
        
        # 分段预测（联合输出）
        outputs = []
        short_len, mid_len, long_len = self.segment_lengths
        if self.head_short is not None:
            outputs.append(self.head_short(ctx))  # [B, short_len, 2]
        if self.head_mid is not None:
            outputs.append(self.head_mid(ctx))  # [B, mid_len, 2]
        if self.head_long is not None:
            outputs.append(self.head_long(ctx))  # [B, long_len, 2]
        
        # 拼接所有分段
        if len(outputs) > 1:
            out = torch.cat(outputs, dim=1)  # [B, horizon, 2]
        else:
            out = outputs[0]  # [B, horizon, 2]
        
        if out.size(1) > self.horizon:
            out = out[:, :self.horizon, :]
        
        return out

# ============ 空间特征增强模块 ============
class subFourierFeatureEncoder(nn.Module):
    """多频率傅里叶编码 - 优化张量操作"""
    def __init__(self, bands=(1, 2, 4, 8, 16)):
        super().__init__()
        self.bands = bands
        # 预计算2π避免重复计算
        self.register_buffer('two_pi', torch.tensor(2 * np.pi))
        
    def forward(self, x):
        """
        x: [..., C] 输入特征 (如 x, y 坐标)
        输出: [..., C*(1+2*len(bands))] 扩展特征
        """
        outputs = [x]
        # 一次性计算所有频带，减少重复操作
        for band in self.bands:
            band_tensor = torch.tensor(band, device=x.device, dtype=x.dtype)
            angle = self.two_pi * band_tensor * x
            outputs.append(torch.sin(angle))
            outputs.append(torch.cos(angle))
        return torch.cat(outputs, dim=-1)


class subResidualMLP(nn.Module):
    """MLP - 优化内存布局"""
    def __init__(self, d_in, d_hidden, horizon, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, horizon)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        # 保持内存连续性
        y = self.drop(self.act(self.fc1(x)))
        y = self.drop(self.act(self.fc2(y)))
        return self.out(y)


class subRelativePositionBias(nn.Module):
    """相对位置偏置 - 优化特征拼接"""
    def __init__(self, n_heads=4):
        super().__init__()
        # 输入维度计算: Fourier(2) + dist + angle + dv + ally = 2*(1+2*4) + 1 + 1 + 2 + 1 = 23
        input_dim = 2 * (1 + 2 * len((1, 2, 4, 8))) + 1 + 1 + 2 + 1
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_heads)
        )
        self.fourier = subFourierFeatureEncoder(bands=(1, 2, 4, 8))
        
    def forward(self, rel_features):
        """
        rel_features: dict with keys 'dxy', 'dist', 'angle', 'dv', 'ally_mask'
        return: [B, T, n_heads, N, N] 加性偏置
        """
        B, T, N, _, _ = rel_features['dxy'].shape
        
        # 优化：展平批处理和时间维度，减少重复计算
        dxy_flat = rel_features['dxy'].reshape(B*T*N*N, 2)
        dxy_fourier = self.fourier(dxy_flat).reshape(B, T, N, N, -1)
        
        # 特征拼接优化：避免多次索引
        features = torch.cat([
            dxy_fourier,
            rel_features['dist'],
            rel_features['angle'],
            rel_features['dv'],
            rel_features['ally_mask']
        ], dim=-1)
        
        bias = self.proj(features)
        return bias.permute(0, 1, 4, 2, 3)


class subFrameSpatialAttention(nn.Module):
    """帧内空间注意力 - 优化掩码和计算"""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # 使用nn.Linear的bias=False减少参数
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rel_bias = subRelativePositionBias(n_heads=n_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # 预计算缩放因子
        self.register_buffer('scale', torch.tensor(self.head_dim ** -0.5))
        
    def forward(self, x, rel_features, mask=None):
        B, T, N, D = x.shape
        
        # 优化：保持内存连续性，避免多次reshape
        q = self.q_proj(x).view(B, T, N, self.n_heads, self.head_dim).transpose(2, 3)
        k = self.k_proj(x).view(B, T, N, self.n_heads, self.head_dim).transpose(2, 3)
        v = self.v_proj(x).view(B, T, N, self.n_heads, self.head_dim).transpose(2, 3)
        
        # 优化：使用einsum提高可读性和性能
        scores = torch.einsum('bthnd,bthmd->bthnm', q, k) * self.scale
        
        # 添加相对位置偏置
        bias = self.rel_bias(rel_features)
        scores = scores + bias
        
        # 优化掩码处理：避免多次类型转换
        if mask is not None:
            # mask: [B, T, N] ->扩展到注意力分数形状
            attn_mask = mask.unsqueeze(2).unsqueeze(3).bool()
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        
        # 优化：使用softmax的dtype参数提高效率
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)
        
        # 优化：使用einsum
        out = torch.einsum('bthnm,bthmd->bthnd', attn, v)
        out = out.transpose(2, 3).reshape(B, T, N, D)
        
        out = self.out_proj(out)
        return self.layer_norm(x + out)


class subSpatioTemporal_GRU(nn.Module):
    """subSpatioTemporal_GRU - 优化RNN和内存使用"""
    def __init__(self, input_dim, horizon, hidden_dim=128, n_spatial_layers=2, 
                 n_temporal_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_temporal_layers = n_temporal_layers
        self.horizon = horizon
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 帧内空间注意力层
        self.spatial_layers = nn.ModuleList([
            subFrameSpatialAttention(hidden_dim, n_heads, dropout)
            for _ in range(n_spatial_layers)
        ])
        
        # 帧间时间编码 - 使用更高效的GRU实现
        self.temporal_gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_temporal_layers,
            batch_first=True, dropout=dropout if n_temporal_layers > 1 else 0
        )
        
        # 预测头
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        # 使用小初始化避免初始预测过大
        self.pool_query = nn.Parameter(torch.randn(1, 2, hidden_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        short_len = min(15, horizon)
        remaining = max(horizon - short_len, 0)
        mid_len = min(15, remaining)
        long_len = max(remaining - mid_len, 0)
        self.segment_lengths = (short_len, mid_len, long_len)
        self.head_short = subResidualMLP(hidden_dim * 2, hidden_dim, short_len) if short_len > 0 else None
        self.head_mid = subResidualMLP(hidden_dim * 2, hidden_dim, mid_len) if mid_len > 0 else None
        self.head_long = subResidualMLP(hidden_dim * 2, hidden_dim, long_len) if long_len > 0 else None
        
    def forward(self, x, rel_features, ego_idx=0, mask=None):
        B, T, N, D = x.shape
        
        # 输入投影 - 保持内存连续
        h = self.input_proj(x).contiguous()
        
        # 帧内空间注意力 - 使用in-place操作减少内存
        for spatial_layer in self.spatial_layers:
            h = spatial_layer(h, rel_features, mask)
        
        # 提取 ego 节点的时序特征
        ego_features = h[:, :, ego_idx, :]  # [B, T, hidden_dim]
        
        # 时间编码 - 优化：h0初始化为None让GRU自动处理
        temporal_out, _ = self.temporal_gru(ego_features)
        
        # 注意力池化 - 优化：保持批处理维度一致
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(temporal_out), self.pool_ln(temporal_out))
        ctx = ctx.reshape(B, -1)
        
        # 预测
        outputs = []
        short_len, mid_len, long_len = self.segment_lengths
        if self.head_short is not None:
            outputs.append(self.head_short(ctx))
        if self.head_mid is not None:
            outputs.append(self.head_mid(ctx))
        if self.head_long is not None:
            outputs.append(self.head_long(ctx))
        out = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
        if out.size(1) > self.horizon:
            out = out[:, :self.horizon]
        # 修改：直接返回 scaled velocity (不进行累积求和)
        # return torch.cumsum(out, dim=1)
        return out

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
        ball_land_x_abs = 0.0
        ball_land_y_abs = 0.0
        
        if 'play_direction_orig' in ego_input.columns:
            play_dir = ego_input['play_direction_orig'].iloc[-1]
        
        if 'ball_land_x_abs' in ego_input.columns:
            ball_land_x_abs = ego_input['ball_land_x_abs'].iloc[-1]
            ball_land_y_abs = ego_input['ball_land_y_abs'].iloc[-1]
        elif 'ball_land_x' in ego_input.columns:
             # Fallback if not transformed (e.g. if running old code path)
            ball_land_x_abs = ego_input['ball_land_x'].iloc[-1]
            ball_land_y_abs = ego_input['ball_land_y'].iloc[-1]
            
        seq_meta.append({
            'game_id': gid,
            'play_id': pid,
            'nfl_id': nid,
            'play_direction': play_dir,
            'ball_land_x_abs': ball_land_x_abs,
            'ball_land_y_abs': ball_land_y_abs,
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
def create_spatiotemporal_subgraphs_ST(input_df, output_df, feature_cols, window_size=10, 
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

if MODE == 'infer':
    # 推理模式：移除控制台处理器，只保留文件日志
    logger_obj = logging.getLogger()
    console_handlers = [h for h in logger_obj.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    for handler in console_handlers:
        logger_obj.removeHandler(handler)
    
    logger.info("="*80)
    logger.info("ENSEMBLE MODE: Loading 3 models with weights [0.4, 0.4, 0.2]")
    logger.info("="*80)
    
    # 初始化全局变量
    models_gru, scalers_gru = [], []
    models_trans, scalers_trans = [], []
    models_sub_x, models_sub_y, scalers_sub = [], [], []
    saved_feature_cols_gru, saved_window_gru, saved_k_neighbors_gru = None, None, None
    saved_feature_cols_trans, saved_window_trans, saved_k_neighbors_trans = None, None, None
    saved_feature_cols_sub, saved_window_sub, saved_k_neighbors_sub = None, None, None
    
    # ============ Model 1: ST-GRU (0.4 weight) ============
    logger.info("[1/3] Loading ST-GRU models...")
    save_dir_gru = "/kaggle/input/st-gru-519"
    save_dir_gru = Path(save_dir_gru)
    meta_path_gru = save_dir_gru / "meta.json"
    if not meta_path_gru.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path_gru}")
    
    with open(meta_path_gru, "r") as f:
        meta_gru = json.load(f)
    
    saved_feature_cols_gru = meta_gru["feature_cols"]
    saved_window_gru = int(meta_gru.get("window_size", WINDOW_SIZE))
    horizon_gru = int(meta_gru.get("max_future_horizon", MAX_FUTURE_HORIZON))
    hidden_dim_gru = int(meta_gru.get("hidden_dim", HIDDEN_DIM))
    saved_k_neighbors_gru = int(meta_gru.get("k_neighbors", K_NEIGHBORS))
    seeds_gru = meta_gru["seeds"]
    n_folds_gru = int(meta_gru["n_folds"])
    
    logger.info(f"  ✓ Meta: {len(saved_feature_cols_gru)} features, window={saved_window_gru}, K={saved_k_neighbors_gru}")
    
    models_gru, scalers_gru = [], []
    for seed in seeds_gru:
        seed_dir = save_dir_gru / f"seed_{seed}"
        for fold in range(1, n_folds_gru + 1):
            sc_path = seed_dir / f"scaler_fold{fold}.pkl"
            model_path = seed_dir / f"model_joint_fold{fold}.pt"
            
            if not (sc_path.exists() and model_path.exists()):
                logger.warning(f"  Missing seed={seed} fold={fold}, skip")
                continue
            
            scaler = joblib.load(sc_path)
            model = SpatioTemporal_GRU(
                len(saved_feature_cols_gru), horizon_gru,
                hidden_dim=hidden_dim_gru,
                n_spatial_layers=2,
                n_temporal_layers=2,
                n_heads=4,
                dropout=0.15
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            scalers_gru.append(scaler)
            models_gru.append(model)
    
    logger.info(f"  ✓ Loaded {len(models_gru)} ST-GRU models (weight=0.4)")
    
    # ============ Model 2: STTransformer (0.4 weight) ============
    logger.info("[2/3] Loading STTransformer models...")
    save_dir_trans = "/kaggle/input/stt-519"
    save_dir_trans = Path(save_dir_trans)
    meta_path_trans = save_dir_trans / "meta.json"
    if not meta_path_trans.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path_trans}")
    
    with open(meta_path_trans, "r") as f:
        meta_trans = json.load(f)
    
    saved_feature_cols_trans = meta_trans["feature_cols"]
    saved_window_trans = int(meta_trans.get("window_size", WINDOW_SIZE))
    horizon_trans = int(meta_trans.get("max_future_horizon", MAX_FUTURE_HORIZON))
    hidden_dim_trans = int(meta_trans.get("hidden_dim", HIDDEN_DIM))
    saved_k_neighbors_trans = int(meta_trans.get("k_neighbors", K_NEIGHBORS))
    seeds_trans = meta_trans["seeds"]
    n_folds_trans = int(meta_trans["n_folds"])
    
    logger.info(f"  ✓ Meta: {len(saved_feature_cols_trans)} features, window={saved_window_trans}, K={saved_k_neighbors_trans}")
    
    models_trans, scalers_trans = [], []
    for seed in seeds_trans:
        seed_dir = save_dir_trans / f"seed_{seed}"
        for fold in range(1, n_folds_trans + 1):
            sc_path = seed_dir / f"scaler_fold{fold}.pkl"
            model_path = seed_dir / f"model_joint_fold{fold}.pt"
            
            if not (sc_path.exists() and model_path.exists()):
                logger.warning(f"  Missing seed={seed} fold={fold}, skip")
                continue
            
            scaler = joblib.load(sc_path)
            model = STransformer(
                len(saved_feature_cols_trans), horizon_trans,
                hidden_dim=hidden_dim_trans,
                n_spatial_layers=2,
                n_temporal_layers=2,
                n_heads=4,
                dropout=0.1,
                window_size=saved_window_trans,
                predict_single_player=True
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            scalers_trans.append(scaler)
            models_trans.append(model)
    
    logger.info(f"  ✓ Loaded {len(models_trans)} STTransformer models (weight=0.4)")
    
    # ============ Model 3: subST-GRU (0.2 weight) ============
    logger.info("[3/3] Loading subST-GRU models...")
    save_dir_sub = "/kaggle/input/model-529"
    save_dir_sub = Path(save_dir_sub)
    meta_path_sub = save_dir_sub / "meta.json"
    if not meta_path_sub.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path_sub}")
    
    with open(meta_path_sub, "r") as f:
        meta_sub = json.load(f)
    
    saved_feature_cols_sub = meta_sub["feature_cols"]
    saved_window_sub = int(meta_sub.get("window_size", WINDOW_SIZE))
    horizon_sub = int(meta_sub.get("max_future_horizon", MAX_FUTURE_HORIZON))
    hidden_dim_sub = int(meta_sub.get("hidden_dim", HIDDEN_DIM))
    saved_k_neighbors_sub = int(meta_sub.get("k_neighbors", K_NEIGHBORS))
    seeds_sub = meta_sub["seeds"]
    n_folds_sub = int(meta_sub["n_folds"])
    
    logger.info(f"  ✓ Meta: {len(saved_feature_cols_sub)} features, window={saved_window_sub}, K={saved_k_neighbors_sub}")
    
    models_sub_x, models_sub_y, scalers_sub = [], [], []
    for seed in seeds_sub:
        seed_dir = save_dir_sub / f"seed_{seed}"
        for fold in range(1, n_folds_sub + 1):
            sc_path = seed_dir / f"scaler_fold{fold}.pkl"
            model_x_path = seed_dir / f"model_x_fold{fold}.pt"
            model_y_path = seed_dir / f"model_y_fold{fold}.pt"
            
            if not (sc_path.exists() and model_x_path.exists() and model_y_path.exists()):
                logger.warning(f"  Missing seed={seed} fold={fold}, skip")
                continue
            
            scaler = joblib.load(sc_path)
            
            model_x = subSpatioTemporal_GRU(
                len(saved_feature_cols_sub), horizon_sub,
                hidden_dim=hidden_dim_sub,
                n_spatial_layers=2,
                n_temporal_layers=2,
                n_heads=4,
                dropout=0.1
            ).to(DEVICE)
            model_x.load_state_dict(torch.load(model_x_path, map_location=DEVICE))
            model_x.eval()
            
            model_y = subSpatioTemporal_GRU(
                len(saved_feature_cols_sub), horizon_sub,
                hidden_dim=hidden_dim_sub,
                n_spatial_layers=2,
                n_temporal_layers=2,
                n_heads=4,
                dropout=0.1
            ).to(DEVICE)
            model_y.load_state_dict(torch.load(model_y_path, map_location=DEVICE))
            model_y.eval()
            
            scalers_sub.append(scaler)
            models_sub_x.append(model_x)
            models_sub_y.append(model_y)
    
    logger.info(f"  ✓ Loaded {len(models_sub_x)} subST-GRU models (weight=0.2)")
    logger.info("="*80)
    logger.info(f"Total models loaded: GRU={len(models_gru)}, Trans={len(models_trans)}, Sub={len(models_sub_x)}")
    logger.info("="*80)

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """推理函数 - 融合三个模型的预测 (权重: 0.4, 0.4, 0.2)"""
    import gc
    
    # 保存原始输入用于 STTransformer (不做相对坐标转换)
    test_input_original = test_input.clone()
    
    # ============ Model 1 & 3: ST-GRU 和 subST-GRU (使用相对坐标) ============
    logger.info("Processing features for ST-GRU and subST-GRU (relative coordinates)...")
    test_input_rel = unify_left_direction(test_input.clone())
    test_input_rel = transform_to_relative_coordinates(test_input_rel)
    
    test_features_rel = engineer_advanced_features(test_input_rel)
    del test_input_rel
    gc.collect()
    
    test_features_rel = add_heading_features(test_features_rel)
    test_features_rel = add_sequence_features(test_features_rel)
    test_features_rel = add_trajectory_and_prediction_features(test_features_rel)
    test_features_rel = add_physics_constraint_features(test_features_rel)
    
    gnn_test_rel = compute_neighbor_embeddings_per_frame(test_features_rel, k_neigh=6, radius=30.0, tau=8.0)
    test_features_rel = test_features_rel.join(gnn_test_rel, on=['game_id', 'play_id', 'frame_id', 'nfl_id'], how='left')
    del gnn_test_rel
    gc.collect()
    
    advanced_graph_test_rel = compute_advanced_graph_features(test_features_rel, k_neigh=6, radius=30.0)
    test_features_rel = test_features_rel.join(advanced_graph_test_rel, on=['game_id', 'play_id', 'nfl_id'], how='left')
    del advanced_graph_test_rel
    gc.collect()
    
    test_features_rel = add_football_rule_features(test_features_rel)
    test_features_rel = add_time_features(test_features_rel)
    test_features_rel = add_qb_relative_features(test_features_rel)
    test_features_rel = clean_features_for_modeling(test_features_rel)
    
    # ============ Model 2: STTransformer (使用原始坐标) ============
    logger.info("Processing features for STTransformer (original coordinates)...")
    test_input_orig = unify_left_direction(test_input_original)
    # 注意：不调用 transform_to_relative_coordinates
    
    test_features_orig = engineer_advanced_features(test_input_orig)
    del test_input_orig
    gc.collect()
    
    test_features_orig = add_heading_features(test_features_orig)
    test_features_orig = add_sequence_features(test_features_orig)
    test_features_orig = add_trajectory_and_prediction_features(test_features_orig)
    test_features_orig = add_physics_constraint_features(test_features_orig)
    
    gnn_test_orig = compute_neighbor_embeddings_per_frame(test_features_orig, k_neigh=6, radius=30.0, tau=8.0)
    test_features_orig = test_features_orig.join(gnn_test_orig, on=['game_id', 'play_id', 'frame_id', 'nfl_id'], how='left')
    del gnn_test_orig
    gc.collect()
    
    advanced_graph_test_orig = compute_advanced_graph_features(test_features_orig, k_neigh=6, radius=30.0)
    test_features_orig = test_features_orig.join(advanced_graph_test_orig, on=['game_id', 'play_id', 'nfl_id'], how='left')
    del advanced_graph_test_orig
    gc.collect()
    
    test_features_orig = add_football_rule_features(test_features_orig)
    test_features_orig = add_time_features(test_features_orig)
    test_features_orig = add_qb_relative_features(test_features_orig)
    test_features_orig = clean_features_for_modeling(test_features_orig)
    
    # ============ 准备模板 ============
    test_template_pl = test.clone()
    
    # ============ Prediction 1: ST-GRU (0.4 weight) ============
    logger.info("[1/3] ST-GRU prediction...")
    for col in saved_feature_cols_gru:
        if col not in test_features_rel.columns:
            test_features_rel = test_features_rel.with_columns([pl.lit(0.0).alias(col)])
    
    test_seqs_gru, test_rel_feats_gru, _, _, test_meta_gru = create_spatiotemporal_subgraphs(
        test_features_rel, test_template_pl, saved_feature_cols_gru,
        window_size=saved_window_gru,
        k_neighbors=saved_k_neighbors_gru,
        cache_path=None,
        show_progress=False
    )
    
    all_preds_gru = []
    for scaler, model in zip(scalers_gru, models_gru):
        X_test_sc = np.stack([
            scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
            for s in test_seqs_gru
        ]).astype(np.float32)
        
        rel_feats = {}
        for key in test_rel_feats_gru[0].keys():
            rel_feats[key] = torch.tensor(
                np.stack([test_rel_feats_gru[i][key] for i in range(len(test_rel_feats_gru))]).astype(np.float32)
            ).to(DEVICE)
        
        X_t = torch.tensor(X_test_sc).to(DEVICE)
        with torch.no_grad():
            pred_joint = model(X_t, rel_feats, ego_idx=0).cpu().numpy()
        all_preds_gru.append(pred_joint)
    
    ens_gru = np.mean(all_preds_gru, axis=0)  # [B, horizon, 2]
    logger.info(f"  ✓ ST-GRU ensemble shape: {ens_gru.shape}")
    
    # ============ Prediction 2: STTransformer (0.4 weight) ============
    logger.info("[2/3] STTransformer prediction...")
    for col in saved_feature_cols_trans:
        if col not in test_features_orig.columns:
            test_features_orig = test_features_orig.with_columns([pl.lit(0.0).alias(col)])
    
    test_seqs_trans, test_rel_feats_trans, _, _, test_meta_trans = create_spatiotemporal_subgraphs_ST(
        test_features_orig, test_template_pl, saved_feature_cols_trans,
        window_size=saved_window_trans,
        k_neighbors=saved_k_neighbors_trans,
        cache_path=None,
        show_progress=False
    )
    
    all_preds_trans = []
    for scaler, model in zip(scalers_trans, models_trans):
        X_test_sc = np.stack([
            scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
            for s in test_seqs_trans
        ]).astype(np.float32)
        
        rel_feats = {}
        for key in test_rel_feats_trans[0].keys():
            rel_feats[key] = torch.tensor(
                np.stack([test_rel_feats_trans[i][key] for i in range(len(test_rel_feats_trans))]).astype(np.float32)
            ).to(DEVICE)
        
        X_t = torch.tensor(X_test_sc).to(DEVICE)
        with torch.no_grad():
            pred_joint = model(X_t, rel_feats, ego_idx=0).cpu().numpy()
        all_preds_trans.append(pred_joint)
    
    ens_trans = np.mean(all_preds_trans, axis=0)  # [B, horizon, 2]
    logger.info(f"  ✓ STTransformer ensemble shape: {ens_trans.shape}")
    
    # ============ Prediction 3: subST-GRU (0.2 weight) ============
    logger.info("[3/3] subST-GRU prediction...")
    for col in saved_feature_cols_sub:
        if col not in test_features_rel.columns:
            test_features_rel = test_features_rel.with_columns([pl.lit(0.0).alias(col)])
    
    test_seqs_sub, test_rel_feats_sub, _, _, test_meta_sub = create_spatiotemporal_subgraphs(
        test_features_rel, test_template_pl, saved_feature_cols_sub,
        window_size=saved_window_sub,
        k_neighbors=saved_k_neighbors_sub,
        cache_path=None,
        show_progress=False
    )
    
    all_preds_sub_x, all_preds_sub_y = [], []
    for scaler, model_x, model_y in zip(scalers_sub, models_sub_x, models_sub_y):
        X_test_sc = np.stack([
            scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
            for s in test_seqs_sub
        ]).astype(np.float32)
        
        rel_feats = {}
        for key in test_rel_feats_sub[0].keys():
            rel_feats[key] = torch.tensor(
                np.stack([test_rel_feats_sub[i][key] for i in range(len(test_rel_feats_sub))]).astype(np.float32)
            ).to(DEVICE)
        
        X_t = torch.tensor(X_test_sc).to(DEVICE)
        with torch.no_grad():
            pred_x = model_x(X_t, rel_feats, ego_idx=0).cpu().numpy()
            pred_y = model_y(X_t, rel_feats, ego_idx=0).cpu().numpy()
        all_preds_sub_x.append(pred_x)
        all_preds_sub_y.append(pred_y)
    
    ens_sub_x = np.mean(all_preds_sub_x, axis=0)  # [B, horizon]
    ens_sub_y = np.mean(all_preds_sub_y, axis=0)  # [B, horizon]
    ens_sub = np.stack([ens_sub_x, ens_sub_y], axis=-1)  # [B, horizon, 2]
    logger.info(f"  ✓ subST-GRU ensemble shape: {ens_sub.shape}")
    
    # 释放特征数据
    del test_features_rel, test_features_orig
    gc.collect()
    
    # ============ 坐标还原（分别处理三个模型） ============
    logger.info("Converting predictions to absolute coordinates...")
    
    # 确保所有预测的 horizon 一致
    min_horizon = min(ens_gru.shape[1], ens_trans.shape[1], ens_sub.shape[1])
    ens_gru = ens_gru[:, :min_horizon, :]
    ens_trans = ens_trans[:, :min_horizon, :]
    ens_sub = ens_sub[:, :min_horizon, :]
    H = min_horizon
    
    # 1. ST-GRU: 相对坐标 -> 绝对坐标（需要加 ball_land）
    logger.info("  [1/3] Processing ST-GRU predictions (relative -> absolute)...")
    ens_gru_vel_x = ens_gru[..., 0] / 10.0
    ens_gru_vel_y = ens_gru[..., 1] / 10.0
    ens_gru_dx = np.cumsum(ens_gru_vel_x, axis=1)
    ens_gru_dy = np.cumsum(ens_gru_vel_y, axis=1)
    
    idx_x_gru = saved_feature_cols_gru.index('x')
    idx_y_gru = saved_feature_cols_gru.index('y')
    
    gru_abs_x = []
    gru_abs_y = []
    for i, meta_row in enumerate(test_meta_gru):
        ball_land_x_abs = meta_row.get('ball_land_x_abs', 0.0)
        ball_land_y_abs = meta_row.get('ball_land_y_abs', 0.0)
        last_x = test_seqs_gru[i][-1, 0, idx_x_gru]
        last_y = test_seqs_gru[i][-1, 0, idx_y_gru]
        
        # 相对坐标 + 球落点 = 绝对坐标（统一方向）
        abs_x = last_x + ens_gru_dx[i] + ball_land_x_abs  # [H]
        abs_y = last_y + ens_gru_dy[i] + ball_land_y_abs  # [H]
        gru_abs_x.append(abs_x)
        gru_abs_y.append(abs_y)
    
    gru_abs_x = np.stack(gru_abs_x, axis=0)  # [B, H]
    gru_abs_y = np.stack(gru_abs_y, axis=0)  # [B, H]
    
    # 2. STTransformer: 原始绝对坐标 -> 绝对坐标（直接累加）
    logger.info("  [2/3] Processing STTransformer predictions (absolute)...")
    ens_trans_vel_x = ens_trans[..., 0] / 10.0
    ens_trans_vel_y = ens_trans[..., 1] / 10.0
    ens_trans_dx = np.cumsum(ens_trans_vel_x, axis=1)
    ens_trans_dy = np.cumsum(ens_trans_vel_y, axis=1)
    
    idx_x_trans = saved_feature_cols_trans.index('x')
    idx_y_trans = saved_feature_cols_trans.index('y')
    
    trans_abs_x = []
    trans_abs_y = []
    for i in range(len(test_seqs_trans)):
        last_x = test_seqs_trans[i][-1, 0, idx_x_trans]
        last_y = test_seqs_trans[i][-1, 0, idx_y_trans]
        
        # 绝对坐标 + 位移 = 新的绝对坐标（统一方向）
        abs_x = last_x + ens_trans_dx[i]  # [H]
        abs_y = last_y + ens_trans_dy[i]  # [H]
        trans_abs_x.append(abs_x)
        trans_abs_y.append(abs_y)
    
    trans_abs_x = np.stack(trans_abs_x, axis=0)  # [B, H]
    trans_abs_y = np.stack(trans_abs_y, axis=0)  # [B, H]
    
    # 3. subST-GRU: 相对坐标 -> 绝对坐标（需要加 ball_land）
    logger.info("  [3/3] Processing subST-GRU predictions (relative -> absolute)...")
    ens_sub_vel_x = ens_sub[..., 0] / 10.0
    ens_sub_vel_y = ens_sub[..., 1] / 10.0
    ens_sub_dx = np.cumsum(ens_sub_vel_x, axis=1)
    ens_sub_dy = np.cumsum(ens_sub_vel_y, axis=1)
    
    idx_x_sub = saved_feature_cols_sub.index('x')
    idx_y_sub = saved_feature_cols_sub.index('y')
    
    sub_abs_x = []
    sub_abs_y = []
    for i, meta_row in enumerate(test_meta_sub):
        ball_land_x_abs = meta_row.get('ball_land_x_abs', 0.0)
        ball_land_y_abs = meta_row.get('ball_land_y_abs', 0.0)
        last_x = test_seqs_sub[i][-1, 0, idx_x_sub]
        last_y = test_seqs_sub[i][-1, 0, idx_y_sub]
        
        # 相对坐标 + 球落点 = 绝对坐标（统一方向）
        abs_x = last_x + ens_sub_dx[i] + ball_land_x_abs  # [H]
        abs_y = last_y + ens_sub_dy[i] + ball_land_y_abs  # [H]
        sub_abs_x.append(abs_x)
        sub_abs_y.append(abs_y)
    
    sub_abs_x = np.stack(sub_abs_x, axis=0)  # [B, H]
    sub_abs_y = np.stack(sub_abs_y, axis=0)  # [B, H]
    
    # ============ 加权融合: 0.4*GRU + 0.4*Trans + 0.2*Sub ============
    logger.info("Weighted ensemble fusion: 0.4*GRU + 0.4*Trans + 0.2*Sub")
    final_abs_x = 0.4 * gru_abs_x + 0.4 * trans_abs_x + 0.2 * sub_abs_x  # [B, H]
    final_abs_y = 0.4 * gru_abs_y + 0.4 * trans_abs_y + 0.2 * sub_abs_y  # [B, H]
    logger.info(f"  ✓ Final ensemble shape: x={final_abs_x.shape}, y={final_abs_y.shape}")
    
    # ============ 生成最终预测 ============
    test_pd = test.to_pandas()
    test_idx = test_pd.set_index(['game_id', 'play_id', 'nfl_id']).sort_index()
    
    rows = []
    for i, meta_row in enumerate(test_meta_gru):
        gid = meta_row['game_id']
        pid = meta_row['play_id']
        nid = meta_row['nfl_id']
        play_dir = meta_row.get('play_direction', None)
        play_is_right = (play_dir == 'right')
        
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
            # 融合后的绝对坐标（统一方向）
            x_uni = np.clip(final_abs_x[i, tt], 0, FIELD_LENGTH)
            y_uni = np.clip(final_abs_y[i, tt], 0, FIELD_WIDTH)
            
            # 转换回原始方向
            x_pred, y_pred = invert_to_original_direction(x_uni, y_uni, play_is_right)
            
            rows.append({
                'x': x_pred,
                'y': y_pred
            })
    
    predictions = pd.DataFrame(rows)
    logger.info(f"✓ Generated {len(predictions)} predictions")
    print(predictions)
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