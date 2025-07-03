# Archive 폴더

이 폴더는 개발 과정에서 생성되었지만 현재 워크플로우에서는 사용하지 않는 파일들을 보관합니다.

## 📁 폴더 구조

### debug_tools/
- `debug_face_similarity.py` - 얼굴 유사도 디버깅 도구 (개발 완료 후 보관)
- `emergency_regroup_faces.py` - 긴급 얼굴 재그룹핑 도구 (정상화 후 보관)

## 🔄 복원 방법

필요시 다음 명령어로 복원할 수 있습니다:

```bash
# 디버그 도구 복원
cp archive/debug_tools/debug_face_similarity.py data_collection/
cp archive/debug_tools/emergency_regroup_faces.py data_collection/
```

## ⚠️ 주의사항

- 이 파일들은 현재 워크플로우와 호환되지 않을 수 있습니다
- 복원 전에 현재 데이터 백업을 권장합니다
- 복원 후 테스트를 반드시 수행하세요

## 🗑️ 삭제 예정

개발이 완전히 안정화되면 이 폴더의 파일들은 삭제될 예정입니다. 