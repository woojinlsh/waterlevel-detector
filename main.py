import os
import time
import io
import json
import requests
import schedule
import cv2  # 💡 RTSP 캡처용 OpenCV 추가
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 로컬 테스트용 환경 변수 로드
load_dotenv()

# ==========================================
# ⚙️ 1. 설정 및 환경 변수 로드
# ==========================================
VERKADA_TOP_LEVEL_API_KEY = os.getenv("VERKADA_TOP_LEVEL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VERKADA_ORG_ID = os.getenv("VERKADA_ORG_ID")               
HELIX_EVENT_TYPE_UID = os.getenv("HELIX_EVENT_TYPE_UID")   

if not all([VERKADA_TOP_LEVEL_API_KEY, GEMINI_API_KEY, VERKADA_ORG_ID, HELIX_EVENT_TYPE_UID]):
    raise ValueError("🚨 필수 환경 변수가 설정되지 않았습니다.")

HELIX_API_URL = f"https://api.verkada.com/cameras/v1/video_tagging/event?org_id={VERKADA_ORG_ID}"

try:
    with open("cameras.json", "r", encoding="utf-8") as f:
        CAMERAS_TO_MONITOR = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("🚨 cameras.json 파일을 찾을 수 없습니다.")

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 🔐 2. Verkada 단기 API Token 발급 (Helix 전송용)
# ==========================================
def get_verkada_token():
    print("🔑 Verkada API Token을 요청합니다...")
    url = "https://api.verkada.com/token"
    headers = {
        "x-api-key": VERKADA_TOP_LEVEL_API_KEY, 
        "accept": "application/json"
    }
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            print("✅ 단기 API Token 발급 성공!")
            return response.json().get('token')
        print(f"❌ 토큰 발급 실패: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ 토큰 요청 오류: {e}")
        return None

# ==========================================
# 📷 3. RTSP 원본 스트림에서 직접 썸네일 캡처
# ==========================================
def get_rtsp_thumbnail(rtsp_url, location):
    print(f"📸 [{location}] RTSP 스트림에서 썸네일을 직접 캡처합니다...")
    try:
        # OpenCV를 사용하여 RTSP 주소로 직접 연결
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print(f"❌ [{location}] RTSP 스트림에 연결할 수 없습니다. (주소나 네트워크를 확인하세요)")
            return None

        # 프레임 한 장 읽어오기
        ret, frame = cap.read()
        cap.release() # 자원 해제 (매우 중요)

        if ret:
            print(f"✅ [{location}] RTSP 이미지 캡처 성공!")
            # OpenCV 이미지를 Gemini가 읽을 수 있는 jpg 바이트 형태로 인코딩
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                return buffer.tobytes()
        
        print(f"❌ [{location}] 영상 프레임을 읽어오지 못했습니다.")
        return None

    except Exception as e:
        print(f"❌ [{location}] RTSP 캡처 중 시스템 오류 발생: {e}")
        return None

# ==========================================
# 🧠 4. Gemini 3.5 Flash 비전 분석
# ==========================================
def analyze_water_level_with_gemini(image_bytes, location):
    model = genai.GenerativeModel('gemini-3.5-flash')
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"❌ 이미지 변환 오류: {e}")
        return None

    prompt = """
    이 사진에는 포천대교 중간 윗부분에 설치된 '노란색 수위 측정기'가 있습니다. 
    수면이 닿은 위치를 파악하여 정확하게 읽어주세요.
    
    [판독 가이드]
    - 숫자는 1~6(m)이며, 작은 눈금 한 칸은 0.1m, 중간 눈금은 0.5m 단위입니다.
    - 소수점 첫째 또는 둘째 자리까지 계산해주세요. (예: 3.4)

    [출력 규칙: 오직 결과문자열 하나만 반환]
    1. 야간이거나 너무 어두운 경우 -> 'NIGHT'
    2. 비가 많이 와서 식별 불가능한 경우 -> 'RAIN'
    3. 기타 이유로 측정 불가인 경우 -> 'UNKNOWN'
    4. 측정 수위가 1미터(숫자 1) 이하인 경우 -> 'UNDER_1M'
    5. 정상 측정 가능한 경우 -> '숫자(예: 3.4)'만 반환
    """
    try:
        response = model.generate_content([prompt, image])
        result_text = response.text.strip()
        print(f"🔍 Gemini 원본 결과: {result_text}")
        
        if result_text == 'NIGHT': return -1.0, "NIGHT"
        if result_text == 'RAIN': return -1.0, "RAIN"
        if result_text == 'UNKNOWN': return -1.0, "UNKNOWN"
        if result_text == 'UNDER_1M': return 1.0, "UNDER_1M"
        
        try:
            return float(result_text), "NORMAL"
        except ValueError:
            print(f"❌ 정의되지 않은 응답 형태입니다: {result_text}")
            return -1.0, "UNKNOWN"
    except Exception as e:
        print(f"❌ Gemini 분석 오류: {e}")
        return None

# ==========================================
# 📤 5. Verkada Helix Event API 전송
# ==========================================
def send_to_verkada_helix(water_level, status, camera_id, location, meas_time, token):
    print(f"📤 [{location}] Helix로 데이터 전송을 시도합니다...")
    payload = {
        "event_type_uid": HELIX_EVENT_TYPE_UID,
        "camera_id": camera_id,
        "time_ms": int(time.time() * 1000),
        "attributes": {
            "camera_id": camera_id,
            "location": location,
            "measurement_time": meas_time,
            "status": status,
            "water_level": water_level
        }
    }
    headers = {
        "content-type": "application/json",
        "x-verkada-auth": token
    }
    try:
        response = requests.post(HELIX_API_URL, json=payload, headers=headers)
        if response.status_code in [200, 201, 202]:
            print(f"✅ [{location}] 데이터 전송 성공! (수위: {water_level}, 상태: {status})")
        else:
            print(f"❌ [{location}] 전송 실패 (상태코드: {response.status_code})")
            print(f"❌ 응답 내용: {response.text}")
    except Exception as e:
        print(f"❌ Helix 전송 오류: {e}")

# ==========================================
# 🔄 6. 메인 스케줄 작업
# ==========================================
def job():
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n--- 🔄 수위 통합 모니터링 시작 ({current_time_str}) ---")
    
    # Helix 전송용 Verkada 토큰만 발급받음
    token = get_verkada_token()
    if not token: 
        print("⚠️ Verkada 토큰 발급 실패로 전체 프로세스를 중단합니다.")
        return
    
    for cam in CAMERAS_TO_MONITOR:
        cam_id = cam.get("camera_id")
        loc = cam.get("location")
        rtsp_url = cam.get("rtsp_url")
        
        print(f"\n📍 작업 대상: {loc}")
        
        if not rtsp_url:
            print(f"⚠️ [{loc}] RTSP URL이 설정되지 않아 건너뜁니다.")
            continue
            
        # 1. RTSP 직접 캡처
        img = get_rtsp_thumbnail(rtsp_url, loc)
        if not img: 
            print(f"⚠️ [{loc}] 이미지가 없어서 Gemini 분석 및 전송을 건너뜁니다.")
            continue
            
        # 2. Gemini 분석
        result = analyze_water_level_with_gemini(img, loc)
        if not result: 
            print(f"⚠️ [{loc}] 분석 결과가 없어서 데이터 전송을 건너뜁니다.")
            continue
            
        level, status = result
        
        # 3. Helix 전송 (원래의 camera_id 사용)
        send_to_verkada_helix(level, status, cam_id, loc, current_time_str, token)
        
        time.sleep(1)

# ==========================================
# ⏰ 실행 (로컬 테스트용 1분 단위)
# ==========================================
if __name__ == "__main__":
    print("🚀 다중 채널 수위 모니터링 앱(RTSP 직접 연결 버전)이 시작되었습니다. (1분 간격 실행)")
    job() 
    
    schedule.every(1).minutes.do(job) 
    
    while True:
        schedule.run_pending()
        time.sleep(1)
