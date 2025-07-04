import json
from pathlib import Path

print(" 간단한 파이프라인 실행 테스트")

domain_path = Path("data/domains/humanoid/face_recognition")

# 1단계: 파일 확인
raw_files = list((domain_path / "1_raw" / "uploads").glob("*.jpg"))
print(f" 1단계: {len(raw_files)}개 원본 파일 발견")

# 2단계: 특이점 추출 시뮬레이션
extracted_path = domain_path / "2_extracted" / "features"
extracted_path.mkdir(exist_ok=True)

trace_data = {}
for raw_file in raw_files:
    feature_filename = f"{raw_file.stem}_f01.jpg"
    feature_path = extracted_path / feature_filename
    feature_path.write_text(raw_file.read_text(encoding="utf-8"), encoding="utf-8")
    trace_data[raw_file.name] = [feature_filename]

print(f" 2단계: {len(raw_files)}개 특이점 추출 완료")

# 3단계: 클러스터링 시뮬레이션
clustered_path = domain_path / "3_clustered" / "groups"
clustered_path.mkdir(exist_ok=True)

group_file = clustered_path / "g001_02.jpg"
group_file.write_text("group data", encoding="utf-8")

print(f" 3단계: 1개 그룹 생성 완료")

# 추적성 정보 업데이트
trace_file = domain_path / "traceability" / "trace.json"
trace_data_full = json.loads(trace_file.read_text(encoding="utf-8"))
trace_data_full["stage_1_to_2"] = trace_data
trace_file.write_text(json.dumps(trace_data_full, indent=2, ensure_ascii=False), encoding="utf-8")

print(" 파이프라인 테스트 완료!")
