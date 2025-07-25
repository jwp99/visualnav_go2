# collect_training_data.py (최종 프로덕션 버전)

import os
import time
import pandas as pd
import cv2
from datetime import datetime
import asyncio
import threading
from queue import Queue, Empty
import logging
import pickle
import numpy as np

# 가상환경에 설치된 라이브러리를 표준 방식으로 바로 임포트합니다.
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC
from aiortc import MediaStreamTrack

# 로그 레벨 설정 (이제 성공했으므로 불필요한 로그는 보지 않습니다)
logging.basicConfig(level=logging.FATAL)

class Go2TrajectoryCollector:
    def __init__(self, robot_ip="192.168.200.157"):
        self.robot_ip = robot_ip
        dataset_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = f"traindata/go2_{dataset_id}"
        self.output_pkl_path = os.path.join(self.dataset_dir, "traj_data.pkl")
        os.makedirs(self.dataset_dir, exist_ok=True)
        print(f"데이터셋이 '{self.dataset_dir}' 폴더에 저장됩니다.")

        self.data_queue = Queue(maxsize=30)
        self.latest_state = {}
        self.state_lock = threading.Lock()
        self.is_running = True
        
        self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.robot_ip)
        self.asyncio_loop = asyncio.new_event_loop()
        self.asyncio_thread = threading.Thread(target=self._run_asyncio_loop, daemon=True)

    def _sportmodestatus_callback(self, message):
        with self.state_lock:
            self.latest_state = message['data']

    async def _recv_camera_stream(self, track: MediaStreamTrack):
        while self.is_running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                
                with self.state_lock:
                    state_at_frame_time = self.latest_state.copy()

                if not self.data_queue.full() and state_at_frame_time:
                    self.data_queue.put((img, state_at_frame_time))
            except Exception:
                break

    async def _async_setup_and_run(self):
        try:
            await self.conn.connect()
            print("✅ 로봇과 성공적으로 연결되었습니다.")
            
            self.conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LF_SPORT_MOD_STATE'], self._sportmodestatus_callback)
            self.conn.video.switchVideoChannel(True)
            self.conn.video.add_track_callback(self._recv_camera_stream)
            
            print("✅ 데이터 수집을 시작합니다. Ctrl+C로 종료하세요.")
            while self.is_running:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"\n!!! 비동기 연결 또는 실행 중 오류 발생: {e} !!!")
            self.is_running = False
        finally:
            if self.conn and self.conn.pc and self.conn.pc.connectionState == "connected":
                await self.conn.disconnect()

    def _run_asyncio_loop(self):
        asyncio.set_event_loop(self.asyncio_loop)
        self.asyncio_loop.run_until_complete(self._async_setup_and_run())

    def start_collection(self):
        self.asyncio_thread.start()
        # 연결이 완료될 때까지 충분히 기다립니다.
        while not (self.conn and self.conn.isConnected):
            time.sleep(1)
            if not self.asyncio_thread.is_alive():
                print("연결 실패. 스크립트를 종료합니다.")
                return

        raw_log_data = []
        image_idx = 0
        try:
            while self.is_running:
                try:
                    image_frame, current_state = self.data_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                if not current_state: continue

                image_filename = f"{image_idx}.jpg"
                image_path = os.path.join(self.dataset_dir, image_filename)
                
                resized_frame = cv2.resize(image_frame, (96, 96))
                cv2.imwrite(image_path, resized_frame)
                image_idx += 1

                # --- 최종 수정된 부분 ---
                # 이제 정확한 경로로 Yaw 값을 가져옵니다.
                yaw_value = current_state['imu_state']['rpy'][2]
                
                log_entry = {
                    'pos_x': current_state['position'][0], 
                    'pos_y': current_state['position'][1],
                    'pos_z': current_state['position'][2], 
                    'yaw': yaw_value
                }
                raw_log_data.append(log_entry)
                print(f"\r수집된 데이터: {len(raw_log_data)}개", end="")
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n'Ctrl + C' 감지. 수집을 종료합니다...")
        
        finally:
            self.is_running = False
            print("\n데이터를 파일로 저장합니다...")
            if raw_log_data:
                df = pd.DataFrame(raw_log_data)
                
                positions = df[['pos_x', 'pos_y']].to_numpy(dtype=np.float64)
                yaws = df['yaw'].to_numpy(dtype=np.float64)

                processed_data = {
                    'position': positions.tolist(),
                    'yaw': yaws.tolist()
                }

                with open(self.output_pkl_path, 'wb') as f:
                    pickle.dump(processed_data, f, protocol=4)
                
                print("\n--- 처리 결과 ---")
                print(f"Position 데이터 형태: {positions.shape}")
                print(f"Yaw 데이터 형태: {yaws.shape}")
                print(f"성공! 처리된 데이터가 '{self.output_pkl_path}'에 저장되었습니다.")
                print("이제 이 폴더를 visualnav-transformer 모델 학습에 사용할 수 있습니다.")

            if self.asyncio_loop.is_running():
                self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
            self.asyncio_thread.join(timeout=5)
            print("프로그램을 종료합니다.")


if __name__ == "__main__":
    collector = Go2TrajectoryCollector(robot_ip="192.168.200.157")
    collector.start_collection()