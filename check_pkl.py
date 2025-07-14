import pickle
import pandas as pd

# pkl 파일 경로 (실제 경로에 맞게 수정하세요)
file_path = 'DB_go_stanford/no1vc_7_0/traj_data.pkl' 

try:
    # 파일을 바이너리 읽기 모드('rb')로 엽니다.
    with open(file_path, 'rb') as f:
        # pickle.load()를 사용해 파일의 내용을 파이썬 객체로 불러옵니다.
        data = pickle.load(f)

    # 불러온 데이터의 타입을 확인합니다. (보통 list 또는 dict)
    print(f"데이터 타입: {type(data)}\n")

    # 데이터가 리스트인 경우
    if isinstance(data, list):
        print(f"총 데이터 개수 (리스트 길이): {len(data)}\n")
        if len(data) > 0:
            print("리스트의 첫 번째 요소:")
            print(data[0])
            print("\n")
            # 리스트 안의 요소가 딕셔너리라면 pandas DataFrame으로 보면 더 편합니다.
            try:
                df = pd.DataFrame(data)
                print("Pandas DataFrame으로 변환된 데이터 (상위 5개):")
                print(df.head())
            except Exception as e:
                print(f"DataFrame 변환 실패: {e}")

    # 데이터가 딕셔너리인 경우
    elif isinstance(data, dict):
        print("딕셔너리의 키들:")
        print(data.keys())
        # 각 키에 해당하는 값의 일부를 확인해볼 수 있습니다.
        for key, value in data.items():
            print(f"\n--- 키 '{key}'의 내용 ---")
            if hasattr(value, 'shape'):
                 print(f"타입: {type(value)}, 형태: {value.shape}")
            elif hasattr(value, '__len__'):
                 print(f"타입: {type(value)}, 길이: {len(value)}")
            print(value)


except FileNotFoundError:
    print(f"에러: 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
except Exception as e:
    print(f"에러 발생: {e}")