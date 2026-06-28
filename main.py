import os
import time
import io
import json
import requests
import schedule
import cv2  # RTSP 캡처용 OpenCV
from google import genai  # 최신 구글 공식 통합 SDK
from PIL import Image
from dotenv import load_dotenv

# 환경 변수 로드 (.env)
load_dotenv()

# ==========================================
# ⚙️ 1. 설정 및 환경 변수 로드
# ==========================================
VERKADA_TOP_LEVEL_API_KEY = os.getenv("VERKADA_TOP_LEVEL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VERKADA_ORG_ID = os.getenv("VERKADA_ORG_ID")               
HELIX_EVENT_TYPE_UID = os.getenv("HELIX_EVENT_TYPE_UID")   

if not all([VERKADA_TOP_LEVEL_API_KEY, GEMINI_API_KEY, VERKADA_ORG_ID, HELIX_EVENT_TYPE_UID]):
    raise ValueError("🚨 필수 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

HELIX_API_URL = f"https://api.verkada.com/cameras/v1/video_tagging/event?org_id={VERKADA_ORG_ID}"

try:
    with open("cameras.json", "r", encoding="utf-8") as f:
        CAMERAS_TO_MONITOR = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("🚨 cameras.json 파일을 찾을 수 없습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 🔐 2. Verkada 단기 API Token 발급
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
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"❌ [{location}] RTSP 스트림 연결 실패.")
            return None

        ret, frame = cap.read()
        cap.release()

        if ret:
            print(f"✅ [{location}] RTSP 이미지 캡처 성공!")
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                return buffer.tobytes()
        
        print(f"❌ [{location}] 영상 프레임을 가져오지 못했습니다.")
        return None
    except Exception as e:
        print(f"❌ [{location}] RTSP 캡처 중 오류: {e}")
        return None

# ==========================================
# 🧠 4. Gemini 3.1 Flash Lite 비전 분석 (프롬프트 파일 연동)
# ==========================================
def get_prompt_text():
    """prompt.txt 파일에서 프롬프트를 읽어옵니다."""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("⚠️ prompt.txt 파일을 찾을 수 없어 기본 내장 프롬프트를 사용합니다.")
        return """이 사진에는 포천대교 중간 윗부분에 설치된 '노란색 수위 측정기'가 있습니다. 
수면이 닿은 위치를 파악하여 정확하게 읽어주세요.
[출력 규칙] 정상일 경우 '숫자(예: 3.4)'만 반환, 그 외 NIGHT, RAIN, UNKNOWN, UNDER_1M 반환"""

def analyze_water_level_with_gemini(image_bytes, location):
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"❌ 이미지 변환 오류: {e}")
        return None

    # 💡 파일에서 프롬프트 불러오기 (핫 리로드 가능)
    prompt = get_prompt_text()
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=[prompt, image]
        )
        result_text = response.text.strip()
        print(f"🔍 Gemini 분석 결과: {result_text}")
        
        if result_text == 'NIGHT': return -1.0, "NIGHT"
        if result_text == 'RAIN': return -1.0, "RAIN"
        if result_text == 'UNKNOWN': return -1.0, "UNKNOWN"
        if result_text == 'UNDER_1M': return 1.0, "UNDER_1M"
        
        try:
            return float(result_text), "NORMAL"
        except ValueError:
            print(f"❌ 정의되지 않은 응답 형태: {result_text}")
            return -1.0, "UNKNOWN"
    except Exception as e:
        print(f"❌ Gemini 분석 중 오류 발생: {e}")
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
        print(f"📦 보낸 데이터: {payload['attributes']}")
        print(f"📨 서버 상세 응답: {response.text}")
        
        if response.status_code in [200, 201, 202]:
            print(f"✅ [{location}] 데이터 전송 성공! (수위: {water_level}, 상태: {status})")
        else:
            print(f"❌ [{location}] 전송 실패 (상태코드: {response.status_code})")
    except Exception as e:
        print(f"❌ Helix 전송 오류: {e}")

# ==========================================
# 🔄 6. 메인 스케줄 작업
# ==========================================
def job():
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n--- 🔄 수위 통합 모니터링 시작 ({current_time_str}) ---")
    
    token = get_verkada_token()
    if not token: return
    
    for cam in CAMERAS_TO_MONITOR:
        cam_id = cam.get("camera_id")
        loc = cam.get("location")
        rtsp_url = cam.get("rtsp_url")
        
        print(f"\n📍 작업 대상: {loc}")
        
        if not rtsp_url: continue
        img = get_rtsp_thumbnail(rtsp_url, loc)
        if not img: continue
            
        result = analyze_water_level_with_gemini(img, loc)
        if not result: continue
            
        level, status = result
        send_to_verkada_helix(level, status, cam_id, loc, current_time_str, token)
        
        time.sleep(1)

# ==========================================
# ⏰ 실행 (1시간 단위 실행)
# ==========================================
if __name__ == "__main__":
    print("🚀 다중 채널 수위 모니터링 앱(최신 SDK + RTSP + 프롬프트 분리)이 시작되었습니다.")
    job() 
    
    schedule.every(1).hours.do(job) 
    
    while True:
        schedule.run_pending()
        time.sleep(1)
