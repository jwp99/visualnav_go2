# process_to_pkl.py

import pandas as pd
import numpy as np
import pickle
import os
import argparse

def process_raw_data(dataset_dir):
    """
    raw_data_log.csv 파일을 읽어 traj_data.pkl 파일로 변환합니다.
    """
    raw_csv_path = os.path.join(dataset_dir, "raw_data_log.csv")
    output_pkl_path = os.path.join(dataset_dir, "traj_data.pkl")

    if not os.path.exists(raw_csv_path):
        print(f"오류: '{raw_csv_path}'를 찾을 수 없습니다. 먼저 데이터 수집을 실행하세요.")
        return

    print(f"'{raw_csv_path}' 파일을 읽어 처리합니다...")
    df = pd.read_csv(raw_csv_path)

    # 예시 데이터셋의 형태와 일치시키기
    # 1. 'position' 키에 해당하는 데이터 추출: (N, 2) 형태의 NumPy 배열
    #    여기서는 2D 평면 위치인 pos_x, pos_y를 사용합니다.
    positions = df[['pos_x', 'pos_y']].to_numpy(dtype=np.float64)

    # 2. 'yaw' 키에 해당하는 데이터 추출: (N,) 형태의 NumPy 배열
    yaws = df['yaw'].to_numpy(dtype=np.float64)

    # 3. 최종 딕셔너리 생성
    processed_data = {
        'position': positions,
        'yaw': yaws
    }
    
    # 4. Pickle 파일로 저장
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print("\n--- 처리 결과 ---")
    print(f"Position 데이터 형태: {processed_data['position'].shape}")
    print(f"Yaw 데이터 형태: {processed_data['yaw'].shape}")
    print(f"성공! 처리된 데이터가 '{output_pkl_path}'에 저장되었습니다.")
    print("이제 이 폴더를 visualnav-transformer 모델 학습에 사용할 수 있습니다.")


if __name__ == '__main__':
    # 터미널에서 데이터셋 폴더 경로를 인자로 받습니다.
    # 사용 예: python process_to_pkl.py go2_dataset_20231027_103000
    parser = argparse.ArgumentParser(description="Go2 원시 데이터를 pkl 파일로 변환합니다.")
    parser.add_argument("dataset_dir", type=str, help="처리할 데이터셋 폴더의 경로")
    args = parser.parse_args()
    
    process_raw_data(args.dataset_dir)